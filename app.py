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
os.environ['ULTRALYTICS_AUTOINSTALL'] = 'False'
os.environ['YOLO_VERBOSE'] = 'False'
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Giới hạn 10MB
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

INFERENCE_LOCK = threading.Lock()

def free_memory():
    """Thu gom rác Python và ép Linux (Render) trả lại RAM phân mảnh về OS"""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

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

# ==============================================================================
# 2. TIỀN XỬ LÝ ẢNH SIÊU NHẸ & GIỮ NGUYÊN TỶ LỆ 4:3
# ==============================================================================
class LetterboxPad:
    """Giữ nguyên tỷ lệ 4:3 (224x168) và bù viền xám (114, 114, 114) cho đủ 224x224"""
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
    """Giải mã ảnh ở mức thấp nhất (draft 256px) và thu nhỏ giữ tỷ lệ về cạnh dài 256px"""
    img = Image.open(file_stream)
    try:
        img.draft('RGB', (256, 256))
    except Exception:
        pass
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (235, 235, 235, 255))
        img = Image.alpha_composite(bg, rgba).convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    if max(img.size) > 256:
        img.thumbnail((256, 256), LANCZOS_RESAMPLE)
    return img

# ==============================================================================
# 3. ROUTE GIAO DIỆN WEB & API
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html', crops=CROPS_INFO)

@app.route('/plant/<crop_type>')
def plant_page(crop_type):
    if crop_type not in CROPS_INFO:
        return "Loại cây không hợp lệ!", 404

    _, mobilenet_path, yolo_path, classes_path, _ = get_crop_paths(crop_type)
    has_yolo = os.path.exists(yolo_path)
    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)

    return render_template(
        'plant.html',
        crop_type=crop_type,
        crop=CROPS_INFO[crop_type],
        has_yolo=has_yolo,
        has_mobilenet=has_mobilenet
    )

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

    return jsonify({
        "classes": classes_info,
        "models": {
            'mobilenet': os.path.exists(mobilenet_path),
            'yolov8': os.path.exists(yolo_path),
            'has_metrics': os.path.exists(metrics_path)
        }
    })

# ==============================================================================
# 4. API CHẨN ĐOÁN TUẦN TỰ (NẠP TỪNG MÔ HÌNH -> ĐOÁN -> XẢ RAM NGAY)
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    _, mobilenet_path, yolo_path, classes_path, _ = get_crop_paths(crop_type)
    has_yolo = os.path.exists(yolo_path)
    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)

    if not has_yolo and not has_mobilenet:
        return jsonify({'error': 'Chưa tìm thấy tệp mô hình trong thư mục models/!'}), 404

    try:
        image = load_rgb_image(file.stream)
    except Exception:
        return jsonify({'error': 'Tệp tải lên không phải hình ảnh hợp lệ!'}), 400

    results_detail = {}

    with INFERENCE_LOCK:
        try:
            # Bước 1: Chạy YOLOv8 xong xả sạch khỏi RAM trước khi nạp MobileNetV2
            if has_yolo:
                try:
                    y_model = YOLO(yolo_path)
                    y_res = y_model(image, imgsz=224, verbose=False)[0]
                    top1_idx = int(y_res.probs.top1)
                    results_detail['yolov8'] = {
                        'disease': y_res.names[top1_idx],
                        'confidence': round(float(y_res.probs.top1conf.item()) * 100, 2)
                    }
                    del y_model, y_res
                except Exception as e:
                    results_detail['yolov8_error'] = str(e)
                finally:
                    free_memory()

            # Bước 2: Chạy MobileNetV2 xong xả sạch khỏi RAM
            if has_mobilenet:
                try:
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

                    tensor = eval_transforms(image).unsqueeze(0)
                    outputs = mb_model(tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, idx = torch.max(probs, 0)

                    results_detail['mobilenet'] = {
                        'disease': class_names[idx.item()],
                        'confidence': round(conf.item() * 100, 2)
                    }
                    del mb_model, tensor, outputs, probs
                except Exception as e:
                    results_detail['mobilenet_error'] = str(e)
                finally:
                    free_memory()
        finally:
            del image
            free_memory()

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
