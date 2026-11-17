import os
import json
import glob
import shutil
import threading
import gc
import random

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐỒ HỌA
# ==============================================================================
# Ép thư mục cấu hình tạm vào /tmp để tránh lỗi phân quyền ghi trên server Linux / Colab
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp')
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

# BẮT BUỘC: Ép Matplotlib dùng backend 'Agg' (không hiển thị giao diện GUI).
# Lý do: Khi chạy PyTorch/YOLO trong Thread ngầm của Flask trên Windows/Linux,
# nếu không có dòng này, Matplotlib sẽ cố mở cửa sổ GUI và làm TREO/CRASH server.
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
# Cho phép nhận Request dữ liệu tối đa 500 MB cho mỗi lần Upload
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
# Tắt ép kiểu UTF-8 sang Unicode ASCII (\uXXXX) để file JSON hiển thị tiếng Việt rõ ràng
app.config['JSON_AS_ASCII'] = False
try:
    app.json.ensure_ascii = False
except AttributeError:
    pass

# Cấu hình thông tin các loại cây trồng hỗ trợ trong hệ thống
CROPS_INFO = {
    'ca-chua': {'name': 'Cà Chua', 'icon': '🍅', 'badge': 'bg-danger'},
    'ca-rot': {'name': 'Cà Rốt', 'icon': '🥕', 'badge': 'bg-warning text-dark'}
}

# ==============================================================================
# 2. HÀM QUẢN LÝ ĐƯỜNG DẪN VÀ TỰ ĐỘNG CHIA TẬP DỮ LIỆU (70 - 20 - 10)
# ==============================================================================
def get_crop_paths(crop_type):
    """
    Tạo & trả về đường dẫn thư mục riêng biệt cho từng loại cây trồng.
    Mục đích: Cách ly dữ liệu và mô hình giữa Cà chua và Cà rốt, tránh ghi đè.
    """
    dataset_dir = os.path.abspath(os.path.join("dataset", crop_type))
    split_dir = os.path.abspath(os.path.join("dataset_split", crop_type))
    models_dir = os.path.abspath("models")
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    mobilenet_path = os.path.join(models_dir, f"{crop_type}_mobilenet.pth")
    yolo_path = os.path.join(models_dir, f"{crop_type}_yolo.pt")
    classes_path = os.path.join(models_dir, f"{crop_type}_classes.json")
    status_path = os.path.join(models_dir, f"{crop_type}_status.json")
    metrics_path = os.path.join(models_dir, f"{crop_type}_metrics.json")
    
    return dataset_dir, split_dir, mobilenet_path, yolo_path, classes_path, status_path, metrics_path

def split_dataset(dataset_dir, split_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    [THẦY HỎI]: Tại sao phải chia tập dữ liệu thành 3 phần Train (70%), Val (20%), Test (10%)?
    [TRẢ LỜI]: 
    - Train (70%): Cho mô hình học và cập nhật trọng số.
    - Val (20%): Thẩm định mô hình trong lúc học để tinh chỉnh siêu tham số và chống Overfitting.
    - Test (10%): Tập dữ liệu hoàn toàn độc lập, chưa từng xuất hiện trong quá trình train, 
                  dùng để đánh giá độ chính xác thực tế cuối cùng.
    """
    # Nếu thư mục phân chia cũ đã tồn tại thì xóa đi làm mới để tránh lẫn lộn dữ liệu cũ
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
        
    splits = ['train', 'val', 'test']
    for s in splits:
        os.makedirs(os.path.join(split_dir, s), exist_ok=True)
        
    split_stats = {'train_total': 0, 'val_total': 0, 'test_total': 0, 'per_class': {}}
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for c in sorted(classes):
        c_src = os.path.join(dataset_dir, c)
        imgs = [f for f in os.listdir(c_src) if os.path.isfile(os.path.join(c_src, f))]
        
        # Cố định Seed ngẫu nhiên (random.seed=42) để kết quả chia tập đồng nhất giữa các lần chạy
        random.seed(42)
        random.shuffle(imgs)
        
        total = len(imgs)
        if total == 0:
            continue
            
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        if n_train == 0 and total > 0: n_train = 1
        if n_val == 0 and total > 1: n_val = 1
        
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]
        
        split_dict = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        
        split_stats['per_class'][c] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs),
            'total': total
        }
        split_stats['train_total'] += len(train_imgs)
        split_stats['val_total'] += len(val_imgs)
        split_stats['test_total'] += len(test_imgs)
        
        # Copy ảnh thực tế vào các thư mục train/val/test riêng biệt
        for s in splits:
            s_dst = os.path.join(split_dir, s, c)
            os.makedirs(s_dst, exist_ok=True)
            for img_name in split_dict[s]:
                shutil.copy2(os.path.join(c_src, img_name), os.path.join(s_dst, img_name))
                
    return split_stats

# ==============================================================================
# 3. HÀM TÍNH TOÁN MA TRẬN NHẦM LẪN (CONFUSION MATRIX) VÀ CHỈ SỐ METRICS
# ==============================================================================
def compute_metrics(y_true, y_pred, class_names):
    """
    [THẦY HỎI]: Công thức tính Precision, Recall, F1-Score và Confusion Matrix là gì?
    [TRẢ LỜI]:
    - Confusion Matrix C[i][j]: Số lượng mẫu thuộc lớp thực tế i nhưng bị AI đoán nhầm thành lớp j.
    - True Positive (TP): Dự đoán đúng lớp bệnh i.
    - False Positive (FP): Lớp khác bị dự đoán nhầm thành lớp i.
    - False Negative (FN): Lớp i bị dự đoán nhầm thành lớp khác.
    - Precision = TP / (TP + FP): Trong các mẫu AI gán nhãn i, bao nhiêu % là đúng thật.
    - Recall = TP / (TP + FN): Trong các mẫu thực tế là i, AI phát hiện được bao nhiêu %.
    - F1-Score = 2 * (Precision * Recall) / (Precision + Recall): Trung bình hài hòa giữa P và R.
    """
    num_classes = len(class_names)
    cm = [[0] * num_classes for _ in range(num_classes)]
    
    # Xây dựng Ma trận nhầm lẫn NxN
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
            
    total_samples = len(y_true)
    if total_samples == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "confusion_matrix": cm, "per_class": {}, "total_samples": 0
        }
        
    # Đường chéo chính sum(cm[i][i]) chính là tổng số mẫu dự đoán ĐÚNG
    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = round((correct / total_samples) * 100, 2)
    
    per_class = {}
    precisions, recalls, f1s = [], [], []
    
    # Tính toán chi tiết cho từng lớp bệnh
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[k][i] for k in range(num_classes) if k != i)
        fn = sum(cm[i][k] for k in range(num_classes) if k != i)
        
        p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
        per_class[name] = {
            "precision": round(p, 2),
            "recall": round(r, 2),
            "f1_score": round(f1, 2),
            "samples": sum(cm[i])
        }
        
    # Macro Average: Trung bình cộng chỉ số của tất cả các lớp
    macro_precision = round(sum(precisions) / num_classes, 2)
    macro_recall = round(sum(recalls) / num_classes, 2)
    macro_f1 = round(sum(f1s) / num_classes, 2)
    
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
        "total_samples": total_samples
    }

def evaluate_mobilenet(model, dataloader, class_names, device):
    """
    Đánh giá MobileNetV2 trên tập dữ liệu được truyền vào (Train/Val/Test).
    [THẦY HỎI]: Tại sao lại dùng torch.no_grad()?
    [TRẢ LỜI]: Để tắt tính năng tính đạo hàm (gradient), giúp tiết kiệm bộ nhớ RAM/VRAM và tăng tốc độ chạy.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            if device != 'cpu':
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return compute_metrics(y_true, y_pred, class_names)

def evaluate_yolo(yolo_model, split_dir, class_names):
    """Đánh giá YOLOv8 trên tập dữ liệu chỉ định"""
    y_true, y_pred = [], []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        c_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(c_dir):
            continue
        gt_idx = class_to_idx[class_name]
        img_files = glob.glob(os.path.join(c_dir, "*.*"))
        for img_path in img_files:
            try:
                res = yolo_model(img_path, verbose=False)[0]
                pred_idx = int(res.probs.top1)
                pred_name = res.names[pred_idx]
                if pred_name in class_to_idx:
                    y_true.append(gt_idx)
                    y_pred.append(class_to_idx[pred_name])
            except Exception:
                pass
    return compute_metrics(y_true, y_pred, class_names)

# ==============================================================================
# 4. QUẢN LÝ TRẠNG THÁI TIẾN TRÌNH & LOG HUẤN LUYỆN
# ==============================================================================
def save_status(crop_type, status, progress, step, log_msg=None, max_logs=200):
    """Lưu tiến trình huấn luyện real-time ra file status.json để giao diện web đọc"""
    _, _, _, _, _, status_path, _ = get_crop_paths(crop_type)
    
    current_data = {"status": status, "progress": progress, "step": step, "logs": []}
    
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                current_data["logs"] = old_data.get("logs", [])
        except Exception:
            pass

    if log_msg:
        current_data["logs"].append(log_msg)
        if len(current_data["logs"]) > max_logs:
            current_data["logs"] = current_data["logs"][-max_logs:]

    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

def read_status(crop_type):
    """Đọc tiến trình huấn luyện từ file JSON"""
    _, _, _, _, _, status_path, _ = get_crop_paths(crop_type)
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "idle", "progress": 0, "step": "Chưa huấn luyện", "logs": []}

def clear_stuck_training_status():
    """Tự động reset trạng thái bị kẹt khi khởi động lại ứng dụng"""
    models_dir = os.path.abspath("models")
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith("_status.json"):
                filepath = os.path.join(models_dir, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get("status") == "running":
                        data["status"] = "idle"
                        data["progress"] = 0
                        data["step"] = "Chưa huấn luyện"
                        data["logs"] = ["⚠️ Trạng thái đã được reset sau khi khởi động server."]
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

clear_stuck_training_status()

# ==============================================================================
# 5. TIỀN XỬ LÝ VÀ CHUẨN HÓA ẢNH (DATA TRANSFORMS)
# ==============================================================================
# [THẦY HỎI]: Tại sao lại dùng Resize(224, 224) và Normalize với các thông số kia?
# [TRẢ LỜI]:
# - Resize (224, 224): Độ phân giải chuẩn đầu vào của mạng MobileNetV2 và YOLOv8.
# - RandomHorizontalFlip, RandomRotation(15): Kỹ thuật Data Augmentation (Tăng cường dữ liệu) 
#   giúp tạo ảnh xoay/lật ngẫu nhiên, chống mô hình học thuộc lòng (Overfitting).
# - Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]): Giá trị Mean và Std chuẩn 
#   của tập dữ liệu ImageNet giúp chuẩn hóa phân phối màu sắc ảnh đầu vào.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_mobilenet_model(num_classes, pretrained=True):
    """
    [THẦY HỎI]: Bản chất việc chỉnh sửa mô hình MobileNetV2 ở đây là gì?
    [TRẢ LỜI]: Sử dụng kỹ thuật Transfer Learning (Học chuyển giao). Nạp trọng số pre-trained 
    từ tập ImageNet (1000 lớp) khi huấn luyện, sau đó thay thế lớp phân loại cuối cùng 
    model.classifier[1] bằng một lớp Linear mới phù hợp với num_classes (số lớp bệnh cây trồng).
    (Khi dự đoán hoặc đánh giá đặt pretrained=False để nạp trực tiếp file trọng số đã train, tiết kiệm RAM).
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# ==============================================================================
# 6. ROUTE ĐIỀU HƯỚNG GIAO DIỆN WEB (FLASK)
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
# 7. CÁC API VỀ TẬP DỮ LIỆU & BÁO CÁO METRICS
# ==============================================================================
@app.route('/api/<crop_type>/dataset-info', methods=['GET'])
def dataset_info(crop_type):
    """Trả về số lượng ảnh của từng lớp bệnh và trạng thái huấn luyện"""
    dataset_dir, split_dir, mobilenet_path, yolo_path, _, _, metrics_path = get_crop_paths(crop_type)
    classes_info = {}
    if os.path.exists(dataset_dir):
        for sub in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, sub)
            if os.path.isdir(sub_path):
                imgs = glob.glob(os.path.join(sub_path, "*.*"))
                classes_info[sub] = len(imgs)
                
    models_status = {
        'mobilenet': os.path.exists(mobilenet_path),
        'yolov8': os.path.exists(yolo_path),
        'has_metrics': os.path.exists(metrics_path)
    }
    
    split_info = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                split_info = data.get("dataset_split_info", {})
        except Exception:
            pass

    return jsonify({"classes": classes_info, "models": models_status, "split_info": split_info})

@app.route('/api/<crop_type>/metrics', methods=['GET'])
def get_metrics(crop_type):
    """API xuất toàn bộ số liệu Confusion Matrix, Precision, Recall, F1 ra giao diện"""
    _, _, _, _, _, _, metrics_path = get_crop_paths(crop_type)
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'error': f'Lỗi đọc số liệu: {str(e)}'}), 500
    return jsonify({'error': 'Chưa có số liệu đánh giá. Vui lòng huấn luyện trước!'}), 404

@app.route('/api/<crop_type>/add-sample', methods=['POST'])
def add_sample(crop_type):
    """API tải lên ảnh mẫu cho các danh mục bệnh"""
    dataset_dir, _, _, _, _, _, _ = get_crop_paths(crop_type)
    disease_name = request.form.get('disease_name', '').strip()
    files = request.files.getlist('files')

    if not disease_name or not files or len(files) == 0:
        return jsonify({'error': 'Vui lòng nhập tên bệnh và chọn ít nhất 1 hình ảnh!'}), 400

    safe_name = disease_name.replace("/", "_").replace("\\", "_")
    class_dir = os.path.join(dataset_dir, safe_name)
    os.makedirs(class_dir, exist_ok=True)

    saved_count = 0
    existing_count = len(os.listdir(class_dir))

    for idx, file in enumerate(files):
        if file and file.filename != '':
            filename = f"{existing_count + idx + 1}.jpg"
            filepath = os.path.join(class_dir, filename)
            try:
                image = Image.open(file.stream).convert('RGB')
                image.save(filepath)
                saved_count += 1
            except Exception:
                continue

    return jsonify({'message': f'Đã tải lên thành công {saved_count} ảnh vào danh mục "{safe_name}"!'})

@app.route('/api/<crop_type>/train-status', methods=['GET'])
def train_status(crop_type):
    return jsonify(read_status(crop_type))

# ==============================================================================
# 8. LUỒNG HUẤN LUYỆN DỰ ÁN & TIẾN TRÌNH ĐÁNH GIÁ (TRAINING WORKER)
# ==============================================================================
@app.route('/api/<crop_type>/train', methods=['POST'])
def train_model(crop_type):
    dataset_dir, split_dir, mobilenet_path, yolo_path, classes_path, status_path, metrics_path = get_crop_paths(crop_type)
    epochs = int(request.form.get('epochs', 5))
    model_choice = request.form.get('model_choice', 'both')
    
    classes = [d for d in os.listdir(dataset_dir) 
               if os.path.isdir(os.path.join(dataset_dir, d)) 
               and len(os.listdir(os.path.join(dataset_dir, d))) > 0]
    
    if len(classes) < 2:
        return jsonify({'error': f'Cần ít nhất 2 nhãn bệnh để AI huấn luyện phân biệt cho {CROPS_INFO[crop_type]["name"]}!'}), 400

    current_status = read_status(crop_type)
    if current_status.get("status") == "running":
        return jsonify({'error': 'Hệ thống đang trong quá trình huấn luyện, vui lòng đợi!'}), 400

    if os.path.exists(status_path):
        try: os.remove(status_path)
        except Exception: pass

    save_status(crop_type, "running", 5, "⚙️ Tự động chia tập Train (70%), Val (20%), Test (10%)...", "⚙️ Công đoạn 1/5: Tự động phân chia tập dữ liệu...")

    def train_worker():
        """
        [THẦY HỎI]: Tại sao lại dùng Threading ở đây?
        [TRẢ LỜI]: Quá trình huấn luyện AI mất nhiều thời gian. Dùng Threading giúp xử lý ngầm, 
        không làm treo giao diện Web Flask, người dùng vẫn có thể xem tiến trình real-time.
        """
        try:
            class_names = sorted(classes)
            with open(classes_path, 'w', encoding='utf-8') as f:
                json.dump(class_names, f, ensure_ascii=False)

            # 1. Tự động chia tập dữ liệu chuẩn 70% - 20% - 10%
            split_stats = split_dataset(dataset_dir, split_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
            save_status(crop_type, "running", 10, 
                        f"📊 Phân chia xong: Train ({split_stats['train_total']}), Val ({split_stats['val_total']}), Test ({split_stats['test_total']})", 
                        f"📊 Đã chia xong: Train={split_stats['train_total']}, Val={split_stats['val_total']}, Test={split_stats['test_total']}")

            has_gpu = torch.cuda.is_available()
            device = 'cuda' if has_gpu else 'cpu'
            yolo_device = 0 if has_gpu else 'cpu'
            
            all_metrics_results = {
                "dataset_split_info": split_stats,
                "class_names": class_names
            }

            # ==========================================
            # A. HUẤN LUYỆN & ĐÁNH GIÁ YOLOV8
            # ==========================================
            if model_choice in ['yolov8', 'both']:
                save_status(crop_type, "running", 15, "⚡ Khởi tạo YOLOv8...", "⚡ Công đoạn 2/5: Khởi tạo YOLOv8 Classification...")

                yolo_model = YOLO('yolov8n-cls.pt')

                def on_yolo_epoch_end(trainer):
                    try:
                        ep = trainer.epoch + 1
                        tot_ep = trainer.epochs
                        end_pct = 45 if model_choice == 'both' else 80
                        pct = 20 + int((ep / tot_ep) * (end_pct - 20))
                        metrics = getattr(trainer, 'metrics', {}) or {}
                        top1 = round(metrics.get('metrics/accuracy_top1', 0) * 100, 2)
                        msg = f"✓ [YOLOv8] Epoch {ep}/{tot_ep} hoàn tất -> Accuracy: {top1}%"
                        save_status(crop_type, "running", pct, f"[YOLOv8] Epoch {ep}/{tot_ep}", msg)
                    except Exception: pass

                yolo_model.add_callback("on_fit_epoch_end", on_yolo_epoch_end)
                yolo_project_dir = os.path.abspath(os.path.join("yolo_runs", crop_type))

                yolo_model.train(
                    data=split_dir, epochs=epochs, imgsz=224, batch=32 if has_gpu else 16,
                    workers=0, device=yolo_device, project=yolo_project_dir, name="train",
                    exist_ok=True, verbose=False, plots=False
                )

                best_weights = os.path.join(yolo_project_dir, "train", "weights", "best.pt")
                if os.path.exists(best_weights):
                    shutil.copy(best_weights, yolo_path)

                # Đánh giá YOLOv8 trên cả 3 tập Train, Val, Test
                save_status(crop_type, "running", 48, "📈 Đánh giá YOLOv8...", "📈 Đang tính toán Confusion Matrix & Metrics cho YOLOv8...")
                yolo_eval_model = YOLO(yolo_path)
                all_metrics_results["yolov8"] = {
                    "train": evaluate_yolo(yolo_eval_model, os.path.join(split_dir, "train"), class_names),
                    "val": evaluate_yolo(yolo_eval_model, os.path.join(split_dir, "val"), class_names),
                    "test": evaluate_yolo(yolo_eval_model, os.path.join(split_dir, "test"), class_names)
                }

                del yolo_model, yolo_eval_model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # ==========================================
            # B. HUẤN LUYỆN & ĐÁNH GIÁ MOBILENETV2
            # ==========================================
            if model_choice in ['mobilenet', 'both']:
                start_p = 50 if model_choice == 'both' else 15
                save_status(crop_type, "running", start_p, "🧠 Nạp MobileNetV2...", "🧠 Công đoạn 3/5: Nạp dữ liệu vào MobileNetV2...")

                train_ds = datasets.ImageFolder(root=os.path.join(split_dir, "train"), transform=train_transforms)
                val_ds = datasets.ImageFolder(root=os.path.join(split_dir, "val"), transform=eval_transforms)
                test_ds = datasets.ImageFolder(root=os.path.join(split_dir, "test"), transform=eval_transforms)

                train_loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=min(32, len(val_ds)), shuffle=False, num_workers=0)
                test_loader = DataLoader(test_ds, batch_size=min(32, len(test_ds)), shuffle=False, num_workers=0)

                tot_batches = len(train_loader)
                mb_model = build_mobilenet_model(len(class_names), pretrained=True)
                if has_gpu: mb_model = mb_model.cuda()

                # [THẦY HỎI]: Hàm Loss function và Optimizer sử dụng ở đây là gì?
                # [TRẢ LỜI]:
                # - Loss function: CrossEntropyLoss (Chuyên dụng cho phân loại đa lớp).
                # - Optimizer: Adam với Learning rate 0.001 giúp tối ưu hóa trọng số hiệu quả.
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(mb_model.parameters(), lr=0.001)

                mb_model.train()
                for epoch in range(epochs):
                    running_loss, correct, total = 0.0, 0, 0

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        if has_gpu: inputs, labels = inputs.cuda(), labels.cuda()

                        # 3 DÒNG HUẤN LUYỆN CỐT LÕI:
                        optimizer.zero_grad()          # 1. Xóa sạch gradient tích lũy từ bước trước
                        outputs = mb_model(inputs)     # 2. Forward pass: Đưa ảnh qua mô hình lấy dự đoán
                        loss = criterion(outputs, labels) # Tính toán độ lỗi (Loss)
                        loss.backward()                # 3. Backpropagation: Lan truyền ngược tính đạo hàm
                        optimizer.step()               # Cập nhật trọng số mô hình

                        running_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()

                        batch_num = batch_idx + 1
                        if batch_num % 5 == 0 or batch_num == tot_batches:
                            current_acc = (correct / total) * 100
                            pct = start_p + int(((epoch + (batch_num / tot_batches)) / epochs) * (85 - start_p))
                            msg = f"[MobileNet] Epoch {epoch+1}/{epochs} | Batch {batch_num}/{tot_batches} -> Loss: {loss.item():.4f} | Acc: {current_acc:.1f}%"
                            save_status(crop_type, "running", pct, f"[MobileNet] Ep {epoch+1}/{epochs}", msg)

                torch.save(mb_model.cpu().state_dict(), mobilenet_path)

                # Đánh giá MobileNetV2 trên cả 3 tập Train, Val, Test
                save_status(crop_type, "running", 88, "📈 Đánh giá MobileNetV2...", "📈 Đang tính toán Confusion Matrix & Metrics cho MobileNetV2...")
                mb_eval_model = build_mobilenet_model(len(class_names), pretrained=False)
                mb_eval_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
                if has_gpu: mb_eval_model = mb_eval_model.cuda()

                all_metrics_results["mobilenet"] = {
                    "train": evaluate_mobilenet(mb_eval_model, train_loader, class_names, device),
                    "val": evaluate_mobilenet(mb_eval_model, val_loader, class_names, device),
                    "test": evaluate_mobilenet(mb_eval_model, test_loader, class_names, device)
                }

                del mb_model, mb_eval_model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # 4. Xuất file JSON chứa toàn bộ dữ liệu báo cáo
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics_results, f, ensure_ascii=False, indent=2)

            save_status(crop_type, "completed", 100, "🎉 Hoàn tất báo cáo số liệu!", f"🎉 HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ CHO {CROPS_INFO[crop_type]['name'].upper()}!")

        except Exception as e:
            save_status(crop_type, "error", 0, "❌ Lỗi huấn luyện", f"❌ Lỗi huấn luyện: {str(e)}")

    threading.Thread(target=train_worker, daemon=True).start()
    return jsonify({'message': 'Đã khởi chạy huấn luyện AI thành công!'})

# ==============================================================================
# 9. API CHẨN ĐOÁN VÀ PHÂN LOẠI ẢNH MỚI (PREDICT)
# ==============================================================================
@app.route('/api/<crop_type>/predict', methods=['POST'])
def predict(crop_type):
    """
    Nhận 1 ảnh người dùng gửi lên, đưa qua cả 2 mô hình (MobileNetV2 & YOLOv8) 
    và tổng hợp kết quả chẩn đoán chính xác nhất.
    """
    if crop_type not in CROPS_INFO:
        return jsonify({'error': 'Loại cây không hợp lệ!'}), 400

    dataset_dir, _, mobilenet_path, yolo_path, classes_path, _, _ = get_crop_paths(crop_type)
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Vui lòng chọn ảnh để chẩn đoán!'}), 400

    has_mobilenet = os.path.exists(mobilenet_path) and os.path.exists(classes_path)
    has_yolo = os.path.exists(yolo_path)

    if not has_mobilenet and not has_yolo:
        return jsonify({'error': f'Chưa có mô hình AI cho {CROPS_INFO[crop_type]["name"]}! Vui lòng huấn luyện trước.'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        # Thu nhỏ ảnh chụp độ phân giải cao từ camera điện thoại về 224x224 ngay từ đầu
        image = image.resize((224, 224))
    except Exception:
        return jsonify({'error': 'File tải lên không phải hình ảnh hợp lệ!'}), 400

    results_detail = {}

    # Dự đoán bằng YOLOv8
    if has_yolo:
        try:
            y_model = YOLO(yolo_path)
            y_res = y_model(image, imgsz=224, verbose=False)[0]
            top1_idx = int(y_res.probs.top1)
            y_disease = y_res.names[top1_idx]
            y_conf = round(float(y_res.probs.top1conf.item()) * 100, 2)
            results_detail['yolov8'] = {'disease': y_disease, 'confidence': y_conf}
            del y_model, y_res
            gc.collect()
        except Exception as e:
            results_detail['yolov8_error'] = str(e)

    # Dự đoán bằng MobileNetV2
    if has_mobilenet:
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)

            mb_model = build_mobilenet_model(len(class_names), pretrained=False)
            mb_model.load_state_dict(torch.load(mobilenet_path, map_location=torch.device('cpu')))
            mb_model.eval()

            tensor = eval_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = mb_model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0) # Dùng Softmax đổi Logits ra % xác suất
                conf, idx = torch.max(probs, 0)

            mb_disease = class_names[idx.item()]
            mb_conf = round(conf.item() * 100, 2)
            results_detail['mobilenet'] = {'disease': mb_disease, 'confidence': mb_conf}
            del mb_model, tensor, outputs, probs
            gc.collect()
        except Exception as e:
            results_detail['mobilenet_error'] = str(e)

    # Đánh giá tổng hợp kết quả giữa 2 mô hình
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
        return jsonify({'error': 'Lỗi khi chạy mô hình AI!', 'details': results_detail}), 500

    return jsonify({
        'disease': final_disease,
        'confidence': final_confidence,
        'details': results_detail
    })

# Khởi chạy máy chủ Flask tại cổng 7860 (Tắt debug & use_reloader để tránh làm hỏng Threading)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
