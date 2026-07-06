import torch
import torchaudio
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from speechbrain.inference.speaker import EncoderClassifier

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "aishell_mini"
OUT_DIR = PROJECT_ROOT / "outputs"

speakers = pd.read_csv(DATA_DIR / "metadata" / "speakers.csv")
utts = pd.read_csv(DATA_DIR / "metadata" / "utterances.csv")

print("num speakers:", speakers["spk_id"].nunique())
print("num utterances:", len(utts))
print('------')
print("gender distribution:")
print(speakers["gender"].value_counts())
print('------')
print("duration statistics:")
print(utts["duration_sec"].describe())

# 检查每个 speaker 的 utterance 数量
spk_counts = utts.groupby("spk_id").size().reset_index(name="num_utts")
spk_counts = spk_counts.merge(speakers, on="spk_id", how="left")

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="LanceaKing/spkrec-ecapa-cnceleb", 
    savedir="ckpt/spkrec-ecapa-cnceleb", 
    run_opts={"device": device})

@torch.no_grad()
def extract_embedding_from_path(path):
    wav, sr = torchaudio.load(path)
    wavs = wav.squeeze(0).unsqueeze(0).to(device)
    emb = classifier.encode_batch(wavs).squeeze().detach().cpu()
    emb = F.normalize(emb, dim=0)
    return emb

# 批量提取 clean embeddings

emb_rows = []
emb_list = []

for _, row in tqdm(utts.iterrows(), total=len(utts)):
    wav_path = DATA_DIR / row["path"]
    emb = extract_embedding_from_path(wav_path)

    emb_list.append(emb.numpy())
    emb_rows.append({
        "utt_id": row["utt_id"],
        "spk_id": row["spk_id"],
        "path": row["path"],
        "condition": "clean",
    })

emb_arr = np.stack(emb_list, axis=0)
emb_meta = pd.DataFrame(emb_rows)

np.save(OUT_DIR / "embeddings" / "embeddings_clean.npy", emb_arr)
emb_meta.to_csv(OUT_DIR / "embeddings" / "embeddings_clean_meta.csv", index=False)

print("embedding array:", emb_arr.shape)

emb_meta_merged = emb_meta.merge(speakers, on="spk_id", how="left")

# t-SNE 可视化函数
# 颜色区分个体、形状区分性别


def plot_tsne(embeddings, meta, title, out_path):
    n = embeddings.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    xy = tsne.fit_transform(embeddings)

    meta = meta.copy()
    meta["x"] = xy[:, 0]
    meta["y"] = xy[:, 1]

    spk_ids = sorted(meta["spk_id"].unique())
    gender_markers = {0: "o", 1: "^", "0": "o", "1": "^"}
    has_multiple_conditions = "condition" in meta.columns and meta["condition"].nunique() > 1

    plt.figure(figsize=(9, 7))

    for spk_id in spk_ids:
        sub_spk = meta[meta["spk_id"] == spk_id]
        group_cols = ["gender", "condition"] if has_multiple_conditions else ["gender"]
        for group_key, sub in sub_spk.groupby(group_cols):
            gender = group_key[0] if has_multiple_conditions else group_key
            condition = group_key[1] if has_multiple_conditions else "clean"
            marker = gender_markers.get(gender, "s")
            alpha = 0.85 if condition == "clean" else 0.35
            gender_name = "Male" if str(gender) == "0" else "Female" if str(gender) == "1" else str(gender)
            label = f"{spk_id}-{gender_name}" if not has_multiple_conditions else f"{spk_id}-{gender_name}-{condition}"
            plt.scatter(
                sub["x"],
                sub["y"],
                marker=marker,
                label=label,
                alpha=alpha,
            )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

    return meta

tsne_clean_meta = plot_tsne(
    emb_arr,
    emb_meta_merged,
    "Clean speaker embeddings t-SNE",
    OUT_DIR / "figures" / "tsne_clean.png",
)

def add_white_noise_snr(wav, snr_db):
    noise = torch.randn_like(wav)

    wav_power = (wav ** 2).mean()
    noise_power = (noise ** 2).mean()

    scale = torch.sqrt(wav_power / (noise_power * (10 ** (snr_db / 10))))
    noisy = wav + scale * noise
    noisy = noisy / (noisy.abs().max() + 1e-8)

    return noisy

# 定义扰动函数二：通道模拟
def simulate_8k_channel(wav):
    down = torchaudio.transforms.Resample(16000, 8000)(wav)
    up = torchaudio.transforms.Resample(8000, 16000)(down)
    up = up / (up.abs().max() + 1e-8)
    return up

# 生成增强音频

AUG_DIR = DATA_DIR / "augmented"
AUG_DIR.mkdir(parents=True, exist_ok=True)

aug_rows = []

for _, row in tqdm(utts.iterrows(), total=len(utts)):
    wav_path = DATA_DIR / row["path"]
    wav, sr = torchaudio.load(wav_path)

    # noise
    noisy = add_white_noise_snr(wav, snr_db=10)
    noisy_rel = f"augmented/noise_snr10/{row['spk_id']}/{row['utt_id']}_noise10.wav"
    noisy_abs = DATA_DIR / noisy_rel
    noisy_abs.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(noisy_abs), noisy.cpu(), 16000)

    aug_rows.append({
        "utt_id": row["utt_id"] + "_noise10",
        "orig_utt_id": row["utt_id"],
        "spk_id": row["spk_id"],
        "path": noisy_rel,
        "condition": "noise_snr10",
    })

    # channel
    ch = simulate_8k_channel(wav)
    ch_rel = f"augmented/channel_8k/{row['spk_id']}/{row['utt_id']}_ch8k.wav"
    ch_abs = DATA_DIR / ch_rel
    ch_abs.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(ch_abs), ch.cpu(), 16000)

    aug_rows.append({
        "utt_id": row["utt_id"] + "_ch8k",
        "orig_utt_id": row["utt_id"],
        "spk_id": row["spk_id"],
        "path": ch_rel,
        "condition": "channel_8k",
    })

aug_df = pd.DataFrame(aug_rows)
aug_df = aug_df.merge(speakers, on="spk_id", how="left")

aug_df.to_csv(DATA_DIR / "metadata" / "augmentations.csv", index=False)

# 批量提取 augmented embeddings
aug_embs = []
aug_meta_rows = []

for _, row in tqdm(aug_df.iterrows(), total=len(aug_df)):
    emb = extract_embedding_from_path(DATA_DIR / row["path"])
    aug_embs.append(emb.numpy())

    aug_meta_rows.append({
        "utt_id": row["utt_id"],
        "orig_utt_id": row["orig_utt_id"],
        "spk_id": row["spk_id"],
        "path": row["path"],
        "condition": row["condition"],
        "gender": row["gender"],
    })

aug_emb_arr = np.stack(aug_embs, axis=0)
aug_meta = pd.DataFrame(aug_meta_rows)

np.save(OUT_DIR / "embeddings" / "embeddings_augmented.npy", aug_emb_arr)
aug_meta.to_csv(OUT_DIR / "embeddings" / "embeddings_augmented_meta.csv", index=False)

print(aug_emb_arr.shape)

combined_emb = np.concatenate([emb_arr, aug_emb_arr], axis=0)

clean_meta_for_plot = emb_meta_merged.copy()
clean_meta_for_plot["condition"] = "clean"
clean_meta_for_plot["orig_utt_id"] = clean_meta_for_plot["utt_id"]

combined_meta = pd.concat(
    [clean_meta_for_plot, aug_meta],
    ignore_index=True,
)

tsne_combined_meta = plot_tsne(
    combined_emb,
    combined_meta,
    "Clean and perturbed speaker embeddings t-SNE",
    OUT_DIR / "figures" / "tsne_clean_augmented_joint.png",
)

# 用 cosine 衡量扰动前后 embedding 漂移量（drift）
clean_lookup = {
    row["utt_id"]: emb_arr[i]
    for i, row in emb_meta.reset_index(drop=True).iterrows()
}

drift_rows = []

for i, row in aug_meta.reset_index(drop=True).iterrows():
    clean_emb = clean_lookup[row["orig_utt_id"]]
    aug_emb = aug_emb_arr[i]

    cos = float(np.dot(clean_emb, aug_emb) / (
        np.linalg.norm(clean_emb) * np.linalg.norm(aug_emb) + 1e-8
    ))

    drift_rows.append({
        "orig_utt_id": row["orig_utt_id"],
        "aug_utt_id": row["utt_id"],
        "spk_id": row["spk_id"],
        "gender": row["gender"],
        "condition": row["condition"],
        "cos_clean_aug": cos,
        "embedding_drift": 1 - cos,
    })

drift_df = pd.DataFrame(drift_rows)

drift_df.to_csv(OUT_DIR / "scores" / "embedding_drift_clean_augmented.csv", index=False)

plt.figure(figsize=(7, 4))
conditions = sorted(drift_df["condition"].unique())
data = [drift_df[drift_df["condition"] == c]["embedding_drift"].values for c in conditions]

plt.boxplot(data, labels=conditions)
plt.ylabel("1 - cosine(clean, perturbed)")
plt.title("Embedding drift under perturbations")
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "embedding_drift_boxplot.png", dpi=150)
plt.show()