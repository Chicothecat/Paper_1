---
description: Bổ sung vào notebook
---

Đối với dữ liệu mang tên DAPT2020
ATTACK_TO_PHASE = {
    # Phase 0 — Benign
    'Benign': 0, 'benign': 0, 'BENIGN': 0, 'Normal': 0, 'normal': 0,

    # Phase 1 — Reconnaissance
    'Network Scan': 1, 'NetworkScan': 1, 'network scan': 1,
    'Web vulnerability Scan': 1, 'Web Vulnerability Scan': 1,
    'WebVulnerabilityScan': 1, 'Port Scan': 1, 'PortScan': 1,
    'Reconnaissance': 1,

    # Phase 2 — Foothold
    'SQL injection': 2, 'SQL Injection': 2, 'SQLInjection': 2,
    'Backdoor': 2, 'backdoor': 2,
    'Malware Download': 2, 'MalwareDownload': 2,
    'Command Injection': 2, 'CommandInjection': 2,
    'CSRF': 2, 'csrf': 2,
    'Account brute force': 2, 'AccountBruteForce': 2, 'BruteForce': 2,
    'Foothold': 2,
    'Establish Foothold': 2,  # <-- THÊM ĐỂ FIX LỖI 8604 DÒNG BỊ LỌT

    # Phase 3 — Lateral Movement
    'DoS': 3, 'dos': 3, 'DenialOfService': 3,
    'Privilege escalation': 3, 'PrivilegeEscalation': 3,
    'Lateral Movement': 3, 'LateralMovement': 3,

    # Phase 4 — Exfiltration
    'Data Exfiltration': 4, 'DataExfiltration': 4,
    'Exfiltration': 4, 'exfiltration': 4,
}

APT_PHASE_NAMES = [
    'Benign', 'Reconnaissance', 'Foothold', 'Lateral Movement', 'Exfiltration'
]
N_APT_PHASES = len(APT_PHASE_NAMES)
# ... (các hằng số còn lại giữ nguyên)


Bổ sung Early Stopping - áp dụng mọi bài
import copy

history = []
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
patience = 7  # Số epoch chờ nếu loss không giảm
patience_counter = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    
    scheduler.step()
    
    # Tính Loss trên tập Validation để làm Early Stopping
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            v_loss = criterion(model(xb), yb)
            val_losses.append(v_loss.item())
            
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    
    m = run_eval(model, val_loader)
    history.append({'ep': ep, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, **{k: m[k] for k in ['acc','f1','prec','rec']}})
    
    print(f'[Epoch {ep}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
          f'Val Acc: {m["acc"]:.4f} | Val F1: {m["f1"]:.4f}')
          
    # Cơ chế Early Stopping & Lưu Checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_STID_model.pth') # Lưu checkpoint
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"🛑 Early stopping kích hoạt tại epoch {ep}! Khôi phục trọng số tốt nhất (Val Loss: {best_val_loss:.4f})")
            break

# KẾT THÚC TRAINING: Load lại trọng số tốt nhất vào mô hình để test
model.load_state_dict(torch.load('best_STID_model.pth'))
print("Đã load checkpoint tốt nhất để chuẩn bị cho tập Test.")

Sửa hàm loss - áp dụng mọi bài 
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        # Cross Entropy KHÔNG nhân weight ở đây
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # Xác suất của lớp ground truth
        p_t = torch.exp(-ce_loss)
        # Focal term: down-weight easy samples
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss
        # Áp dụng class weight
        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights
        return loss.mean()

# Nghịch đảo tần suất chuẩn hóa (theo bài báo)
class_counts = np.bincount(y_tr, minlength=N_APT_PHASES)
inverse_freq = 1.0 / (class_counts + 1e-6)          # tránh chia 0
normalized_weights = inverse_freq / np.sum(inverse_freq)  # tổng = 1
class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(DEVICE)
criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)
