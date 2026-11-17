import os
import json
import gc

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG CHO MÔI TRƯỜNG CLOUD (RENDER)
# ==============================================================================
# Ép thư mục cấu hình tạm vào /tmp để tránh lỗi phân quyền ghi trên server Linux[cite: 1]
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

import matplotlib
matplotlib.use('Agg')  # Tắt giao diện đồ họa GUI tránh treo server

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Xác định chính xác thư mục gốc đang chứa file app.py này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Giới hạn ảnh tải lên 50MB
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

# Danh sách các loại cây trồng hỗ trợ trong hệ thống
CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

# ==============================================================================
# 2. HÀM LẤY ĐƯỜNG DẪN TỆP MÔ HÌNH ĐÃ TRAIN (NẰM CÙNG CẤP VỚI app.py)
# ==============================================================================
def get_crop_paths(crop_type):
    """
    Trỏ thẳng tới các tệp đã huấn luyện sẵn trong thư mục models/ nằm cạnh app.py[cite: 1, 2]:
    - {crop_type}_mobilenet.pth : Trọng số MobileNetV2
    - {crop_type}_yolo.pt       : Trọng số YOLOv8
    - {crop_type}_classes.json  : Danh sách tên các loại bệnh
    - {crop_type}_metrics.json  : Bảng điểm đánh giá (Confusion Matrix, F1, Accuracy)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    dataset_dir = os.path.join(DATASET_DIR, crop_type)
    mobilenet_path = os.path.join(MODELS_DIR, f"{crop_type}_mobilenet.pth")
    yolo_path = os.path.join(MODELS_DIR, f"{crop_type}_yolo.pt")
    classes_path = os.path.join(MODELS_DIR, f"{crop_type}_classes.json")
    metrics_path = os.path.join(MODELS_DIR, f"{crop_type}_metrics.json")

    return dataset_dir, mobilenet_path, yolo_path, classes_path, metrics_path

# ==============================================================================
# 3. TIỀN XỬ LÝ ẢNH & KHỞI TẠO KHUNG MÔ HÌNH MOBILENETV2
# ==============================================================================
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_rgb_image(file_stream):
    """
    Đọc ảnh chuẩn xác: tự động xoay đúng chiều EXIF và xử lý nền trong suốt (PNG/WebP)
    thành nền xám sáng thay vì nền đen kịt khi tải ảnh từ mạng.
    """
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (235, 235, 235, 255))
        img = Image.alpha_composite(bg, rgba).convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def build_mobilenet_model(num_classes):
    """
    Khởi tạo khung MobileNetV2 rỗng (weights=None) để nạp trực tiếp tệp .pth đã train,
    không tải lại trọng số từ Internet giúp tiết kiệm RAM tối đa trên Render.
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# ==============================================================================
# 4. ROUTE GIAO DIỆN WEB (FLASK)[cite: 1, 2]
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html', crops=CROPS_INFO)

@app.route('/plant/<crop_type>')
def plant_page(crop_type):
    if crop_type not in CROPS_INFO:
        return "Loại cây không hợp lệ!", 404
    return render_template('plant.html', crop_type=crop_type, crop=CROPS_INFO[crop_type])

# ==============================================================================
# 5. API KIỂM TRA TRẠNG THÁI MÔ HÌNH & BÁO CÁO METRICS CÓ SẴN
# ==============================================================================
@app.route('/api/<crop_type>/dataset-info', methods=['GET'])
def dataset_info(crop_type):
    """Kiểm tra các tệp mô hình đã có sẵn trong thư mục models/ hay chưa[cite: 1, 2]"""
    dataset_dir, mobilenet_path, yolo_path, classes_path, metrics_path = get_crop_paths(crop_type)

    classes_info = {}
    if os.path.exists(dataset_dir):
        for sub in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, sub)
            if os.path.isdir(sub_path):
                classes_info[sub] = len(os.listdir(sub_path))
    elif os.path.exists(classes_path):
        # Nếu không đẩy thư mục dataset lên Render, lấy danh sách tên bệnh từ file classes.json
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
    """Đọc số liệu đánh giá có sẵn từ file {crop_type}_metrics.json[cite: 1, 2]"""
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
    """Trả về trạng thái sẵn sàng để giao diện plant.html không bị lỗi gọi API"""
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
# 6. API CHẨN ĐOÁN BỆNH TỪ TỆP ĐÃ TRAIN (PREDICT)[cite: 1, 2]
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    """
    Nạp tệp trọng số từ thư mục models/ (nằm cùng cấp với app.py)
    và chạy dự đoán trên ảnh người dùng tải lên[cite: 1, 2].
    """
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    _, mobilenet_path, yolo_path, classes_path, _ = get_crop_paths(crop_type)
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)
    has_yolo = os.path.exists(yolo_path)

    if not has_mobilenet and not has_yolo:
        return jsonify({
            'error': f'Không tìm thấy tệp mô hình trong thư mục models/ cho {CROPS_INFO[crop_type]["name"]}!'
        }), 404

    try:
        image = load_rgb_image(file.stream)
    except Exception:
        return jsonify({'error': 'Tệp tải lên không phải hình ảnh hợp lệ (tránh định dạng .avif)!'}), 400

    results_detail = {}

    # 1. Chẩn đoán bằng YOLOv8 từ tệp models/{crop_type}_yolo.pt[cite: 1, 2]
    if has_yolo:
        try:
            y_model = YOLO(yolo_path)
            y_res = y_model(image, verbose=False)[0]
            top1_idx = int(y_res.probs.top1)
            y_disease = y_res.names[top1_idx]
            y_conf = round(float(y_res.probs.top1conf.item()) * 100, 2)
            results_detail['yolov8'] = {'disease': y_disease, 'confidence': y_conf}
            del y_model, y_res
            gc.collect()
        except Exception as e:
            results_detail['yolov8_error'] = str(e)

    # 2. Chẩn đoán bằng MobileNetV2 từ tệp models/{crop_type}_mobilenet.pth[cite: 1, 2]
    if has_mobilenet:
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

            mb_model = build_mobilenet_model(len(class_names))
            mb_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
            mb_model.eval()

            tensor = eval_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = mb_model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)

            mb_disease = class_names[idx.item()]
            mb_conf = round(conf.item() * 100, 2)
            results_detail['mobilenet'] = {'disease': mb_disease, 'confidence': mb_conf}
            del mb_model, tensor, outputs, probs
            gc.collect()
        except Exception as e:
            results_detail['mobilenet_error'] = str(e)

    # 3. Tổng hợp kết quả chẩn đoán[cite: 1, 2]
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
