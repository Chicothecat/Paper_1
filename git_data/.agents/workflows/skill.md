---
description: Chuyển đổi và tái cấu trúc pipeline huấn luyện NIDS/Botnet sang chuẩn K-Fold Cross-Validation chống Data 
---

---
name: nids-kfold-pipeline-refactor
description: Chuyển đổi và tái cấu trúc pipeline huấn luyện NIDS/Botnet sang chuẩn K-Fold Cross-Validation chống Data Leakage. Sử dụng khi người dùng cần refactor code Jupyter Notebook hoặc script PyTorch từ Hold-out (70/30) sang Stratified K-Fold kết hợp tiền xử lý, feature selection (Extra-Trees) và chuẩn hóa bên trong từng fold.
---

# NIDS K-Fold Cross-Validation Pipeline Refactor

Chuyên môn hóa việc chuyển đổi kiến trúc thực nghiệm phát hiện xâm nhập mạng (NIDS) trên dữ liệu luồng mạng (CTU-13, CIC-IDS) từ phân chia tĩnh (Hold-out) sang **Stratified K-Fold Cross-Validation** chặt chẽ về mặt học thuật.

## Nguyên tắc cốt lõi (Tránh rò rỉ dữ liệu - Data Leakage)
1. **Chia Fold trước, Fit xử lý sau**: Tuyệt đối không tính toán thống kê (Median, Standard Deviation), không chạy Feature Selection (`ExtraTreesClassifier`), và không fit `StandardScaler` trên toàn bộ tập dữ liệu trước vòng lặp.
2. **Khép kín trong Fold**: Mọi thao tác `fit` chỉ được thực hiện trên `X_train` của fold hiện tại. `X_val` và `X_test` chỉ được dùng phương thức `transform`.
3. **Cô lập trạng thái mô hình**: Mỗi fold phải khởi tạo đối tượng mô hình mới (reset weights), `optimizer`, `scheduler`, và `GradScaler` mới để tránh rò rỉ trọng số giữa các fold.
4. **Tách 3 tập cục bộ**: Mỗi fold chia thành:
   - `test_idx`: Dùng để đánh giá độc lập cuối fold (Test Fold).
   - `train_idx`: Cắt thêm một phần nhỏ (ví dụ 10%) làm `val_idx` phục vụ Early Stopping / lưu Best Checkpoint, phần còn lại dùng huấn luyện.

---

## Các bước triển khai chi tiết

### 1. Chuẩn hóa dữ liệu thô (Cell cấu hình & đọc dữ liệu)
- **Đường dẫn & Cấu hình**: Khai báo đường dẫn file CSV/Parquet, số lượng fold (`K_FOLDS = 5`), số đặc trưng cần chọn (`N_SELECT = 20`), và nhãn nhị phân (`CLASS_NAMES = ['Benign', 'Botnet']`).
- **Làm sạch đặc thù**:
  - Loại bỏ các cột định danh mạng và thời gian: `StartTime`, `SrcAddr`, `DstAddr`, `sTos`, `dTos`, `Label`.
  - Mã hóa các cột chuỗi dạng phân loại: `Proto`, `Dir`, `State` bằng `LabelEncoder`.
  - Chuyển đổi kiểu số, xử lý giá trị khuyết (`fillna(0)`), loại bỏ các cột hằng số (`std == 0`).
  - Chuẩn hóa nhãn nhị phân: Gán `1` nếu chứa từ khóa tấn công (`Botnet`/`Attack`), ngược lại gán `0` (`Benign`).

### 2. Thiết kế vòng lặp K-Fold
Khởi tạo `StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)`. Duyệt qua từng fold:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import torch

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*20} FOLD {fold + 1}/{skf.n_splits} {'='*20}")
    
    # Cắt tập dữ liệu thô của fold
    X_tr_val_raw, y_tr_val = X.iloc[train_val_idx], y[train_val_idx]
    X_te_raw, y_te = X.iloc[test_idx], y[test_idx]
    
    # Tách validation set nội bộ từ train_val_idx
    # (Tỷ lệ 90% train / 10% val)
    
    # Bước A: Xử lý ngoại lai & thống kê (Fit trên train, Transform trên val/test)
    # Bước B: Feature Selection (Extra-Trees fit trên train, Transform trên val/test)
    # Bước C: StandardScaler (Fit trên train, Transform trên val/test)
    
    # Bước D: Đóng gói PyTorch Dataset & DataLoader
    # Bước E: Khởi tạo mô hình mới, Optimizer, Loss, Scheduler
    # Bước F: Vòng lặp Train / Validate (Lưu best model)
    # Bước G: Đánh giá trên X_test của fold -> Tính Acc, Prec, Rec, F1