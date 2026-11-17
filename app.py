import os
import sys
import json
import gc
import ctypes
import threading

# ==============================================================================
# 1. GIỚI HẠN CPU & RAM TỐI ĐA CHO RENDER (PHẢI ĐẶT TRƯỚC KHI IMPORT TORCH/YOLO)
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Khóa 1 luồng CPU và tắt hoàn toàn tính đạo hàm (Autograd) để tiết kiệm RAM
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
torch.set_grad_enabled(False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LANCZOS_RESAMPLE = getattr(Image, 'Resampling', Image).LANCZOS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Giới hạn ảnh 20MB tránh tràn RAM
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

# ==============================================================================
# 2. HÀM GIẢI PHÓNG RAM HỆ THỐNG & BỘ NHỚ ĐỆM 1 MÔ HÌNH DUY NHẤT
# ==============================================================================
def free_memory():
    """Thu gom rác Python và ép Linux (Render) trả lại RAM phân mảnh về OS"""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

MODEL_CACHE = {
    "crop_type": None,
    "yolo": None,
    "mobilenet": None,
    "class_names": None
}
INFERENCE_LOCK = threading.Lock()

def get_crop_paths(crop_type):
    os.makedirs(MODELS_DIR, exist_ok=True)
    dataset_dir = os.path.join(DATASET_DIR, crop_type)
    mobilenet_path = os.path.join(MODELS_DIR, f"{crop_type}_mobilenet.pth")
    yolo_path = os.path.join(MODELS_DIR, f"{crop_type}_yolo.pt")
    classes_path = os.path.join(MODELS_DIR, f"{crop_type}_classes.json")
    metrics_path = os.path.join(MODELS_DIR, f"{crop_type}_metrics.json")
    return dataset_dir, mobilenet_path, yolo_path, classes_path, metrics_path

def build_mobilenet_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_cached_models(crop_type):
    """Nạp mô hình của crop_type (nếu đang giữ cây khác trong RAM thì xả sạch trước)"""
    if MODEL_CACHE["crop_type"] == crop_type:
        return MODEL_CACHE["yolo"], MODEL_CACHE["mobilenet"], MODEL_CACHE["class_names"]

    MODEL_CACHE["crop_type"] = None
    MODEL_CACHE["yolo"] = None
    MODEL_CACHE["mobilenet"] = None
    MODEL_CACHE["class_names"] = None
    free_memory()

    _, mobilenet_path, yolo_path, classes_path, _ = get_crop_paths(crop_type)

    y_model = None
    if os.path.exists(yolo_path):
        y_model = YOLO(yolo_path)

    mb_model = None
    class_names = None
    if os.path.exists(mobilenet_path) and os.path.exists(classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        mb_model = build_mobilenet_model(len(class_names))
        try:
            state_dict = torch.load(mobilenet_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(mobilenet_path, map_location='cpu')
        mb_model.load_state_dict(state_dict)
        mb_model.eval()
        del state_dict

    MODEL_CACHE["crop_type"] = crop_type
    MODEL_CACHE["yolo"] = y_model
    MODEL_CACHE["mobilenet"] = mb_model
    MODEL_CACHE["class_names"] = class_names
    free_memory()

    return y_model, mb_model, class_names

# ==============================================================================
# 3. TIỀN XỬ LÝ GIỮ NGUYÊN TỶ LỆ 4:3 (KHÔNG CẮT RÌA, KHÔNG BÓP MÉO)
# ==============================================================================
class LetterboxPad:
    """
    Giữ trọn 100% khung hình 4:3 (hoặc 16:9, 3:4):
    Thu nhỏ theo đúng tỷ lệ về cạnh dài 224px (224x168) và đệm viền xám chuẩn (114, 114, 114)
    để đạt đúng kích thước 224x224 mà không làm biến dạng hình ảnh hay cắt mất rìa lá.
    """
    def __init__(self, size=224, fill=(114, 114, 114)):
        self.size = (size, size)
        self.fill = fill

    def __call__(self, img):
        return ImageOps.pad(img, self.size, method=LANCZOS_RESAMPLE, color=self.fill)

eval_transforms = transforms.Compose([
    LetterboxPad(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_rgb_image(file_stream):
    """Đọc ảnh, xoay đúng hướng EXIF và thu nhỏ giữ nguyên tỷ lệ (cạnh dài tối đa 640px) để nhẹ RAM"""
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (235, 235, 235, 255))
        img = Image.alpha_composite(bg, rgba).convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # thumbnail() luôn giữ nguyên tỷ lệ gốc (ví dụ 4000x3000 -> 640x480)
    if max(img.size) > 640:
        img.thumbnail((640, 640), LANCZOS_RESAMPLE)
    return img

# ==============================================================================
# 4. ROUTE GIAO DIỆN WEB & API
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html', crops=CROPS_INFO)

@app.route('/plant/<crop_type>')
def plant_page(crop_type):
    if crop_type not in CROPS_INFO:
        return "Loại cây không hợp lệ!", 404
    return render_template('plant.html', crop_type=crop_type, crop=CROPS_INFO[crop_type])

@app.route('/api/<crop_type>/dataset-info', methods=['GET'])
def dataset_info(crop_type):
    dataset_dir, mobilenet_path, yolo_path, classes_path, metrics_path = get_crop_paths(crop_type)

    classes_info = {}
    if os.path.exists(dataset_dir):
        for sub in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, sub)
            if os.path.isdir(sub_path):
                classes_info[sub] = len(os.listdir(sub_path))
    elif os.path.exists(classes_path):
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                for c_name in json.load(f):
                    classes_info[c_name] = 0
        except Exception:
            pass

    models_status = {
        'mobilenet': os.path.exists(mobilenet_path),
        'yolov8': os.path.exists(yolo_path),
        'has_metrics': os.path.exists(metrics_path)
    }

    split_info = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                split_info = json.load(f).get("dataset_split_info", {})
        except Exception:
            pass

    return jsonify({"classes": classes_info, "models": models_status, "split_info": split_info})

@app.route('/api/<crop_type>/metrics', methods=['GET'])
def get_metrics(crop_type):
    _, _, _, _, metrics_path = get_crop_paths(crop_type)
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'error': f'Lỗi đọc số liệu: {str(e)}'}), 500
    return jsonify({'error': 'Chưa tìm thấy tệp metrics.json trong thư mục models/!'}), 404

@app.route('/api/<crop_type>/train-status', methods=['GET'])
def train_status(crop_type):
    _, mobilenet_path, yolo_path, _, _ = get_crop_paths(crop_type)
    ready = os.path.exists(mobilenet_path) or os.path.exists(yolo_path)
    return jsonify({
        "status": "completed" if ready else "idle",
        "progress": 100 if ready else 0,
        "step": "Sẵn sàng chẩn đoán (Đã nạp mô hình)" if ready else "Chưa có tệp mô hình trong models/",
        "logs": []
    })

@app.route('/api/<crop_type>/add-sample', methods=['POST'])
def add_sample(crop_type):
    return jsonify({'error': 'Bản triển khai trên Render chỉ dùng để chẩn đoán từ mô hình đã huấn luyện sẵn!'}), 400

@app.route('/api/<crop_type>/train', methods=['POST'])
def train_model(crop_type):
    return jsonify({'error': 'Bản triển khai trên Render đã khóa tính năng Train để tiết kiệm RAM. Vui lòng dùng mô hình có sẵn!'}), 400

# ==============================================================================
# 5. API CHẨN ĐOÁN BỆNH (ĐÃ TỐI ƯU KHÓA LUỒNG & GIẢI PHÓNG RAM SAU MỖI LƯỢT)
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    try:
        image = load_rgb_image(file.stream)
    except Exception:
        return jsonify({'error': 'Tệp tải lên không phải hình ảnh hợp lệ (tránh định dạng .avif)!'}), 400

    results_detail = {}

    with INFERENCE_LOCK:
        try:
            y_model, mb_model, class_names = get_cached_models(crop_type)

            if y_model is None and mb_model is None:
                return jsonify({
                    'error': f'Không tìm thấy tệp mô hình trong thư mục models/ cho {CROPS_INFO[crop_type]["name"]}!'
                }), 404

            # 1. Dự đoán bằng YOLOv8 (tự động letterbox giữ nguyên tỷ lệ 4:3)
            if y_model is not None:
                try:
                    y_res = y_model(image, imgsz=224, verbose=False)[0]
                    top1_idx = int(y_res.probs.top1)
                    y_disease = y_res.names[top1_idx]
                    y_conf = round(float(y_res.probs.top1conf.item()) * 100, 2)
                    results_detail['yolov8'] = {'disease': y_disease, 'confidence': y_conf}
                    del y_res
                except Exception as e:
                    results_detail['yolov8_error'] = str(e)

            # 2. Dự đoán bằng MobileNetV2 (dùng LetterboxPad giữ nguyên tỷ lệ 4:3)
            if mb_model is not None and class_names:
                try:
                    tensor = eval_transforms(image).unsqueeze(0)
                    outputs = mb_model(tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, idx = torch.max(probs, 0)

                    mb_disease = class_names[idx.item()]
                    mb_conf = round(conf.item() * 100, 2)
                    results_detail['mobilenet'] = {'disease': mb_disease, 'confidence': mb_conf}
                    del tensor, outputs, probs
                except Exception as e:
                    results_detail['mobilenet_error'] = str(e)
        finally:
            del image
            free_memory()

    # Tổng hợp kết quả chẩn đoán
    final_disease, final_confidence = "", 0.0

    if 'yolov8' in results_detail and 'mobilenet' in results_detail:
        y_data, m_data = results_detail['yolov8'], results_detail['mobilenet']
        if y_data['disease'] == m_data['disease']:
            final_disease = y_data['disease']
            final_confidence = round((y_data['confidence'] + m_data['confidence']) / 2, 2)
        else:
            if y_data['confidence'] >= m_data['confidence']:
                final_disease, final_confidence = y_data['disease'], y_data['confidence']
            else:
                final_disease, final_confidence = m_data['disease'], m_data['confidence']
    elif 'yolov8' in results_detail:
        final_disease, final_confidence = results_detail['yolov8']['disease'], results_detail['yolov8']['confidence']
    elif 'mobilenet' in results_detail:
        final_disease, final_confidence = results_detail['mobilenet']['disease'], results_detail['mobilenet']['confidence']
    else:
        return jsonify({'error': 'Lỗi khi chạy mô hình chẩn đoán!', 'details': results_detail}), 500

    return jsonify({
        'disease': final_disease,
        'confidence': final_confidence,
        'details': results_detail
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
