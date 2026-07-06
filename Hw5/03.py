from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "aishell_mini"
OUT_DIR = PROJECT_ROOT / "outputs"

enroll_df = pd.read_csv(DATA_DIR / "protocols" / "enroll.csv")
id_test_df = pd.read_csv(DATA_DIR / "protocols" / "identification_test.csv")
utterances_df = pd.read_csv(DATA_DIR / "metadata" / "utterances.csv")
dev_trials = pd.read_csv(DATA_DIR / "protocols" / "verification_trials_dev.csv")
test_trials = pd.read_csv(DATA_DIR / "protocols" / "verification_trials_test.csv")

print("num enrollment speakers:", enroll_df["spk_id"].nunique())
print("num identification test utterances:", len(id_test_df))
print("dev trial labels:")
print(dev_trials["label"].value_counts())



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

# 构建 embedding cache
embedding_cache = {}

def get_embedding_by_path(path):
    path = str(path)
    if path not in embedding_cache:
        embedding_cache[path] = extract_embedding_from_path(DATA_DIR / path)
    return embedding_cache[path]

# 构建 enrollment templates
speaker_templates = {}

for spk_id, group in enroll_df.groupby("spk_id"):
    embs = []

    for _, row in group.iterrows():
        emb = get_embedding_by_path(row["path"])
        embs.append(emb)

    template = np.mean(embs, axis=0)

    speaker_templates[spk_id] = template

print("num templates:", len(speaker_templates))
print("template dim:", next(iter(speaker_templates.values())).shape)

# 示例：计算 cosine score（打分），以一个 positive trial 和一个 negative trial 为例
# same-speaker trial 和 different-speaker trial 的打分方法是一样的，只是在保存文件时要记录ground truth。
def cosine_score(emb1, emb2):
    emb1 = F.normalize(emb1, dim=0)
    emb2 = F.normalize(emb2, dim=0)
    return torch.dot(emb1, emb2).item()

lookup = utterances_df.set_index("utt_id")

pos_trial = dev_trials[dev_trials["label"] == 1].iloc[0]
neg_trial = dev_trials[dev_trials["label"] == 0].iloc[0]

for name, trial in [("positive", pos_trial), ("negative", neg_trial)]:
    enroll_spk = trial["enroll_spk_id"]
    test_utt = trial["test_utt_id"]

    enroll_emb = speaker_templates[enroll_spk]
    test_path = lookup.loc[test_utt, "path"]
    test_emb = get_embedding_by_path(test_path)

    enroll_emb = torch.tensor(enroll_emb)
    test_emb = torch.tensor(test_emb)

    score = cosine_score(enroll_emb, test_emb)

    print(name, trial["trial_id"], "score =", score, "label =", trial["label"])
    
# veritication trial 批量打分
lookup = utterances_df.set_index("utt_id")

def score_trials(trials_df):
    rows = []

    for _, trial in trials_df.iterrows():
        enroll_spk = trial["enroll_spk_id"]
        test_utt = trial["test_utt_id"]

        enroll_emb = speaker_templates[enroll_spk]
        test_path = lookup.loc[test_utt, "path"]
        test_emb = get_embedding_by_path(test_path)

        enroll_emb = torch.tensor(enroll_emb)
        test_emb = torch.tensor(test_emb)

        score = cosine_score(enroll_emb, test_emb)

        rows.append({
            **trial.to_dict(),
            "score": score,
        })

    return pd.DataFrame(rows)

dev_scores = score_trials(dev_trials)
test_scores = score_trials(test_trials)

dev_scores.to_csv(OUT_DIR / "scores" / "verification_dev_scores.csv", index=False)
test_scores.to_csv(OUT_DIR / "scores" / "verification_test_scores.csv", index=False)

# 在 dev trials 上计算 dev EER 最低时的阈值 tau，用于 test trials 正式测试
def compute_far_frr(scores, labels, thresholds):
    rows = []
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)

    pos = labels == 1
    neg = labels == 0

    for tau in thresholds:
        pred_same = scores >= tau

        far = ((pred_same == 1) & neg).sum() / max(neg.sum(), 1)
        frr = ((pred_same == 0) & pos).sum() / max(pos.sum(), 1)

        rows.append({
            "threshold": tau,
            "FAR": far,
            "FRR": frr,
            "abs_diff": abs(far - frr),
        })

    return pd.DataFrame(rows)

thresholds = np.linspace(dev_scores["score"].min(), dev_scores["score"].max(), 300)
dev_curve = compute_far_frr(dev_scores["score"], dev_scores["label"], thresholds)

best = dev_curve.iloc[dev_curve["abs_diff"].argmin()]
tau = best["threshold"]
eer = 0.5 * (best["FAR"] + best["FRR"])

print("Dev tau:", tau)
print("Dev FAR:", best["FAR"])
print("Dev FRR:", best["FRR"])
print("Dev EER:", eer)

# 绘制 dev trials 上的直方图
same_scores = dev_scores[dev_scores["label"] == 1]["score"]
diff_scores = dev_scores[dev_scores["label"] == 0]["score"]

from matplotlib import pyplot as plt

plt.figure(figsize=(7, 4))
plt.hist(same_scores, bins=20, alpha=0.6, label="same speaker")
plt.hist(diff_scores, bins=20, alpha=0.6, label="different speakers")
plt.axvline(tau, linestyle="--", label="dev threshold")
plt.xlabel("cosine score")
plt.ylabel("count")
plt.title("Verification score distribution on dev trials")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "verification_score_hist_dev.png", dpi=150)
plt.show()

# 绘制 dev trials 上的FAR/FRR曲线
plt.figure(figsize=(7, 4))
plt.plot(dev_curve["threshold"], dev_curve["FAR"], label="FAR")
plt.plot(dev_curve["threshold"], dev_curve["FRR"], label="FRR")
plt.axvline(tau, linestyle="--", label="approx EER threshold")
plt.xlabel("threshold")
plt.ylabel("rate")
plt.title("FAR / FRR on dev trials")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "far_frr_curve_dev.png", dpi=150)
plt.show()