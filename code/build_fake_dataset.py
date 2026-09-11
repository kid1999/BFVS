import pandas as pd
import random
import copy
import numpy as np

# fix random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def signed_log_transform(chunk):
    """
    - 保留网络流的方向性 (正数为上行，负数为下行)
    - 缩放包大小的极值影响，防止大包主导距离计算
    """
    arr = np.array(chunk, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))

def generate_sa_cutmix_samples(
    df,
    feature_col="features",
    label_col="label",
    unknown_label=-1,
    min_replace_ratio=0.2,
    max_replace_ratio=0.5,
    num_candidates=100,  # 构建候选池的大小
    top_k_select=3,      # 在最相似的 Top-K 中随机选，保证多样性
    seed=42,
):
    """
    基于语义对齐和最相似子序列替换的高质量未知流量生成器
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. 构建全量样本的类别池
    label_to_samples = {}
    for _, row in df.iterrows():
        label_to_samples.setdefault(row[label_col], []).append(row[feature_col])

    labels = list(label_to_samples.keys())
    fake_samples = []

    for _, row in df.iterrows():
        label = row[label_col]
        features = row[feature_col]
        n = len(features)

        if n <= 1:
            continue

        # 2. 动态决定 CutMix 的窗口大小和位置
        lam = random.uniform(min_replace_ratio, max_replace_ratio)
        k = max(2, int(n * lam))
        k = min(k, n - 1)
        start_idx = random.randint(0, n - k)

        # 提取被挖去的"原序列块"，并转换到 Signed-Log 空间
        orig_chunk = features[start_idx : start_idx + k]
        orig_transformed = signed_log_transform(orig_chunk)

        # 3. 构建异类候选池
        other_labels = [l for l in labels if l != label]
        if not other_labels:
            continue
            
        candidate_chunks = []
        for _ in range(num_candidates):
            donor_label = random.choice(other_labels)
            donor_sample = random.choice(label_to_samples[donor_label])
            donor_len = len(donor_sample)
            
            # 提取长度相同的候选块
            if donor_len >= k:
                donor_start = random.randint(0, donor_len - k)
                chunk = donor_sample[donor_start : donor_start + k]
            else:
                chunk = (donor_sample * (k // donor_len + 1))[:k]
            
            candidate_chunks.append(chunk)

        # 4. 在流形空间中计算距离，寻找最相似序列
        distances = []
        for cand in candidate_chunks:
            cand_transformed = signed_log_transform(cand)
            # 计算欧氏距离 (Frobenius norm)
            dist = np.linalg.norm(orig_transformed - cand_transformed)
            distances.append(dist)

        # 5. 选取最相似的候选块
        distances = np.array(distances)
        # 找到距离最小的 top_k 个索引
        top_k_indices = np.argsort(distances)[:top_k_select]
        
        # 从最相似的几个中随机挑一个
        chosen_idx = random.choice(top_k_indices)
        best_donor_chunk = candidate_chunks[chosen_idx]

        # 6. 执行 CutMix 缝合
        new_features = copy.deepcopy(features)
        new_features[start_idx : start_idx + k] = best_donor_chunk

        fake_samples.append({
            feature_col: new_features,
            label_col: unknown_label,
            "num_flows": len(new_features),
            "mix_ratio": round(k / n, 3),
            "swap_distance": distances[chosen_idx] 
        })

    return fake_samples

if __name__ == "__main__":
    # 示例运行
    df = pd.read_pickle("/data/wf/udfs/data/baidu20_0817.pkl")
    print(f"Original samples: {len(df)}")
    
    # 这里的 num_candidates
    res = generate_sa_cutmix_samples(df, num_candidates=100, top_k_select=10)
    fake_df = pd.DataFrame(res)
    
    print(f"Generated Unknown samples: {len(fake_df)}")
    fake_df.to_pickle("/data/wf/udfs/data/baidu20_fake_200.pkl")