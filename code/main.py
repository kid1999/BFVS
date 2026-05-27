import math
import os
import pickle
import random
from typing import List
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

# =========================
# 0. Set Random Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 已知类别的训练集
TRAIN_PKL = "/data/wf/bfvs/data/github10_20250303.pkl"
# 合成的负样本数据集 (用于作为 N+1 类进行训练) - 请修改此路径
NEGATIVE_TRAIN_PKL = "/data/wf/bfvs/data/github10_20250303_fake.pkl"
# 测试集 (包含已知和未知)
TEST_PKL = "/data/wf/bfvs/data/github10_20250502.pkl"
# 额外的测试用未知样本 (可选)
WORLD_PKL = "/data/wf/bfvs/data/github100_raw.pkl" 

# --- Log and Model Output ---
SAVE_DIR = "/data/wf/bfvs/output/github10_20250303"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "model.pt")
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

# Training Hyperparameters
LR = 1e-4
EPOCHS = 300
BATCH_SIZE = 64

# Transformer Model Hyperparameters
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1


# =========================
# 2. Data Loading and Preprocessing
# =========================
def log_normalize(seq):
    arr = np.array(seq, dtype=np.float32)
    # 保持符号的 log 归一化
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    return torch.tensor(arr, dtype=torch.float32)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels


class FlowDataset(Dataset):
    def __init__(self, X: List[torch.Tensor], y: List[str], le: LabelEncoder, known_class_set: set):
        """
        N+1 策略的核心 Dataset
        """
        self.X = X
        self.le = le
        self.known_class_set = known_class_set

        # 定义未知类别的索引 = 已知类别数量
        # 例如已知 0..49 (共50类), 则未知类索引为 50
        self.unknown_index = len(self.le.classes_)

        y_mapped = []
        for l in y:
            if l in self.known_class_set:
                # 已知样本：正常编码 (0 ~ N-1)
                y_mapped.append(le.transform([l])[0])
            else:
                # 未知/负样本：映射为 N
                y_mapped.append(self.unknown_index)

        self.y = torch.tensor(y_mapped, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.dim() == 1:
            x = torch.stack([x, torch.zeros_like(x)], dim=-1)
        return x, self.y[idx]


# =========================
# 3. Transformer Model (N+1 Softmax)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class FlowTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layer_norm = nn.LayerNorm(d_model)

        # === 修改点：增加分类头 ===
        # 输出维度为 num_classes (这里传入的值已经是 N+1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        seq_mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
        src_key_padding_mask = torch.cat([cls_mask, seq_mask], dim=1)

        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = output[:, 0, :]
        final_feature = self.layer_norm(cls_output)

        # === 修改点：直接返回 Logits ===
        logits = self.classifier(final_feature)
        return logits


# =========================
# 4. Training and Evaluation Function
# =========================
def train_eval_n_plus_1(train_loader, test_loader, num_known_classes, le, device):
    # 总类别数 = 已知类数量 + 1 (Unknown类)
    total_classes = num_known_classes + 1

    model = FlowTransformer(
        input_dim=2, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        num_classes=total_classes
    ).to(device)

    # 使用标准交叉熵，将Unknown视为普通的一个类别
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(
        f"\nModel initialized for {total_classes} classes (0-{num_known_classes - 1}: Known, {num_known_classes}: Unknown)")
    
    # --- 训练阶段耗时统计初始化 ---
    train_start_time = time.perf_counter()
    total_train_samples_processed = 0

    # -------- Training --------
    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for X_batch, lengths, y_batch in pbar:
            X_batch, lengths, y_batch = X_batch.to(device), lengths.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch, lengths)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_samples_processed += len(y_batch) # 累加所有 epoch 的样本总数

            total_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"[Epoch {epoch + 1}/{EPOCHS}] Avg Loss={total_loss / max(1, n_samples):.6f}")
        
    # --- 结束训练计时 ---
    train_end_time = time.perf_counter()
    total_train_duration = train_end_time - train_start_time
    avg_train_time_per_sample = total_train_duration / max(1, total_train_samples_processed)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    def per_label_f1_dict(y_true, y_pred, labels=None, zero_division=0):
        # 所有原始 label 名 + Unknown
        target_names = list(le.classes_) + ["Unknown"]

        # 确定需要计算 F1 的标签集
        labels = np.unique(np.concatenate([y_true, y_pred]))

        # 计算 F1
        f1s = f1_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=zero_division
        )

        # 将整数标签解码为原始类别名
        label_names = []
        for l in labels:
            if l < len(le.classes_):
                label_names.append(le.inverse_transform([l])[0])
            else:
                label_names.append("Unknown")

        return dict(zip(label_names, f1s))

    # -------- Evaluation --------
    # --- 推理阶段耗时统计初始化 ---
    infer_start_time = time.perf_counter()
    total_infer_samples_processed = 0
    
    model.eval()
    y_true_all = []
    y_pred_all = []    # N+1 类的直接预测结果
    all_logits = []    # 保存原始输出用于计算 AUROC 和 强制闭集预测
    
    with torch.no_grad():
        for X_batch, lengths, y_batch in tqdm(test_loader, desc="Evaluation"):
            X_batch, lengths = X_batch.to(device), lengths.to(device)
            
            logits = model(X_batch, lengths)
            preds = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu()) # 转移到CPU保存
            y_true_all.extend(y_batch.numpy())
            y_pred_all.extend(preds.cpu().numpy())
            total_infer_samples_processed += len(y_batch) # 累加测试集样本数
            
    
    # --- 结束推理计时 ---
    infer_end_time = time.perf_counter()
    total_infer_duration = infer_end_time - infer_start_time
    avg_infer_time_per_sample = total_infer_duration / max(1, total_infer_samples_processed)

    # 转换为 Tensor/Array 处理
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    all_logits = torch.cat(all_logits, dim=0) # Shape: [N_samples, num_classes + 1]
    
    # 获取 Softmax 概率 (用于 AUROC)
    all_probs = torch.softmax(all_logits, dim=1).numpy()
    
    # ==========================================
    # 1. 封闭集性能分析 (Closed-Set Performance)
    # ==========================================
    print("\n" + "="*40)
    print(" 1. Closed-Set Performance (Known Samples Only)")
    print("="*40)
    
    # 筛选掩码：真实标签不是未知类的样本
    known_mask = y_true_all != num_known_classes
    
    if np.sum(known_mask) > 0:
        y_true_closed = y_true_all[known_mask]
        
        # 获取这些样本对应的 Logits
        logits_closed = all_logits[known_mask]
        
        # 核心：切片操作 logits_closed[:, :num_known_classes]
        # 含义：强制模型在已知类别中选一个概率最大的，完全忽略“未知”选项
        preds_closed_forced = torch.argmax(logits_closed[:, :num_known_classes], dim=1).numpy()
        
        acc_closed = accuracy_score(y_true_closed, preds_closed_forced)
        pre_closed = precision_score(y_true_closed, preds_closed_forced, average="macro", zero_division=0)
        rec_closed = recall_score(y_true_closed, preds_closed_forced, average="macro", zero_division=0)
        f1_closed  = f1_score(y_true_closed, preds_closed_forced, average="macro", zero_division=0)
        
        print(f"Test Samples (Known): {len(y_true_closed)}")
        print(f"Accuracy  : {acc_closed:.4f}")
        print(f"Precision : {pre_closed:.4f} (Macro)")
        print(f"Recall    : {rec_closed:.4f} (Macro)")
        print(f"F1 Score  : {f1_closed:.4f} (Macro)")
    else:
        print("No known samples in test set! Skipping closed-set evaluation.")
        
        
    # f1_closed = per_label_f1_dict(
    #     y_true_closed,
    #     preds_closed_forced,
    #     zero_division=0
    # )
    
    # print(f1_closed)


    # ==========================================
    # 2. 开放集性能指标 (Open-Set Performance)
    # ==========================================
    print("\n" + "="*50)
    print(" 2. Open-Set Overall Performance (Macro Avg of N+1 classes)")
    print("="*50)
    
    # --- A. 整体多分类指标 (N+1 Classes) ---
    # 定义：对 N+1 个类别计算 Accuracy, Precision, Recall, F1
    # 这里的 Accuracy 就是通常所说的 OS-ACC
    
    os_acc = accuracy_score(y_true_all, y_pred_all)
    os_pre = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    os_rec = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    os_f1  = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    
    print(f"OS-Accuracy  : {os_acc:.4f}")
    print(f"OS-Precision : {os_pre:.4f} (Macro)")
    print(f"OS-Recall    : {os_rec:.4f} (Macro)")
    print(f"OS-F1        : {os_f1:.4f}  (Macro)")

    # --- B. AUROC and TPR@FPR95 (排序与拒识能力) ---
    # 构建二分类标签 (0: Known, 1: Unknown)
    y_true_binary = (y_true_all == num_known_classes).astype(int)
    # 获取 Softmax 输出中“未知类”那一列的概率
    y_score_unknown = all_probs[:, num_known_classes]
    
    try:
        # Calculate AUROC
        auroc = roc_auc_score(y_true_binary, y_score_unknown)
        
        # Calculate ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score_unknown)
        
        # Find the max TPR where FPR <= 0.05 (TPR@FPR=5%)
        # This tells us how many unknowns we catch while allowing a 5% false alarm rate on knowns.
        valid_idx = np.where(fpr <= 0.05)[0]
        tpr_at_fpr95 = tpr[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
        
        print(f"OS-AUROC     : {auroc:.4f} (Known vs Unknown)")
        print(f"TPR@FPR95    : {tpr_at_fpr95:.4f}")
    except ValueError:
        print("OS-AUROC     : Error (Likely only one class present in test set)")
        print("TPR@FPR95    : Error")
        

    # ==========================================
    # 3. 未知类检测详解 (Binary: Known vs Unknown)
    # ==========================================
    print("\n" + "="*50)
    print(" 3. Unknown Detection Performance (Binary: Known=0, Unknown=1)")
    print("="*50)
    
    # 将预测结果二值化：如果预测是 0~N-1 -> 0 (Known); 如果预测是 N -> 1 (Unknown)
    y_pred_binary = (y_pred_all == num_known_classes).astype(int)
    
    # 计算二分类指标
    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    bin_pre = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    bin_rec = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    bin_f1  = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    
    print(f"Detection Accuracy  : {bin_acc:.4f}")
    print(f"Detection Precision : {bin_pre:.4f} (Precision of Unknown)")
    print(f"Detection Recall    : {bin_rec:.4f} (Recall of Unknown / TPR)")
    print(f"Detection F1        : {bin_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix:")
        print(f"                 Pred Known (0)   Pred Unknown (1)")
        print(f"True Known (0)   {tn:<14}   {fp:<14} (FPR: {fp/(fp+tn):.4f})")
        print(f"True Unknown (1) {fn:<14}   {tp:<14} (TPR: {tp/(tp+fn):.4f})")
    
    # 打印完整的分类报告供参考
    target_names = list(le.classes_) + ["Unknown"]
    unique_labels = sorted(list(set(y_true_all) | set(y_pred_all)))
    valid_names = [target_names[i] for i in unique_labels]
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true_all, y_pred_all, labels=unique_labels, target_names=valid_names, zero_division=0))
    
    
    # ==========================================
    # --- 新增：耗时统计输出 ---
    # ==========================================
    print("\n" + "=" * 50)
    print(" 4. Latency Analysis (Time Profiling)")
    print("=" * 50)
    print(f"Total Training Time       : {total_train_duration:.2f} s (Processed {total_train_samples_processed} sample passes across {EPOCHS} epochs)")
    print(f"Avg Train Time per Sample : {avg_train_time_per_sample * 1000:.4f} ms")
    print(f"Total Inference Time      : {total_infer_duration:.2f} s (Processed {total_infer_samples_processed} samples)")
    print(f"Avg Infer Time per Sample : {avg_infer_time_per_sample * 1000:.4f} ms")
    


# =========================
# 5. Main Workflow
# =========================
def main():
    """
    主函数，用于执行整个训练和评估流程
    包括数据加载、预处理、模型训练和评估等步骤
    """
    # 1. 加载已知类别的训练数据
    if not os.path.exists(TRAIN_PKL):
        print(f"Error: {TRAIN_PKL} not found.")
        return
    train_df = pd.read_pickle(TRAIN_PKL)  # 从pickle文件加载训练数据
    
    if TEST_PKL and os.path.exists(TEST_PKL):
        test_df = pd.read_pickle(TEST_PKL)  # 如果存在测试集文件，则加载测试数据
    else:
        print('no test set.')
        train_df, test_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df['label']
        )
    
    # 2. 加载负样本数据 (未知类别)
    if os.path.exists(NEGATIVE_TRAIN_PKL):
        neg_df = pd.read_pickle(NEGATIVE_TRAIN_PKL)
        print(f"Loaded Negative Data: {len(neg_df)} samples.")

        # 强制设置负样本标签为 -1
        neg_df["label"] = "unknown"

        # 合并训练集
        combined_train_df = pd.concat([train_df, neg_df], ignore_index=True)
    else:
        print(f"Warning: Negative data {NEGATIVE_TRAIN_PKL} not found! Training only on known classes (Closed Set).")
        combined_train_df = train_df

    # 修复：增加对 WORLD_PKL 是否为空的判断
    if WORLD_PKL and os.path.exists(WORLD_PKL):
        world_df = pd.read_pickle(WORLD_PKL)
        world_df["label"] = "unknown"  # 确保测试集的未知样本标签为 -1
        combined_test_df = pd.concat([test_df, world_df], ignore_index=True)
    else:
        combined_test_df = test_df

    print(f"Final Train Size: {len(combined_train_df)}")
    print(f"Final Test Size:  {len(combined_test_df)}")

    # 4. Fit Label Encoder (只针对已知类别)
    # 🚨 修复：从更新后的 combined_train_df 中提取，防止之前的 train_df 变量已经失效或不包含负样本切分后的逻辑
    valid_labels = combined_train_df[
        combined_train_df["label"] != "unknown"
    ]["label"].values

    if os.path.exists(LABEL_ENCODER_PATH):
        # 建议每次重新生成，防止pkl里的encoder和当前数据不一致
        pass

    le = LabelEncoder()
    le.fit(valid_labels)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    known_class_set = set(le.classes_)
    print(f"Known Classes: {len(known_class_set)}")

    # 5. Data Loaders
    # Dataset 会自动把不在 known_class_set 里的(即 -1) 映射为 第 N 类
    X_train = [log_normalize(f) for f in combined_train_df["features"]]
    X_test = [log_normalize(f) for f in combined_test_df["features"]]

    train_loader = DataLoader(
        FlowDataset(X_train, combined_train_df["label"].values, le, known_class_set),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        FlowDataset(X_test, combined_test_df["label"].values, le, known_class_set),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 6. Train & Eval
    train_eval_n_plus_1(train_loader, test_loader, num_known_classes=len(known_class_set), le=le, device=device)

    print("\nAll tasks completed.")





if __name__ == "__main__":
    main()
    

