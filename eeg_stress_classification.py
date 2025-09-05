
"""
EEG-Signal-Classification-for-Stress-Level-Detection (Synthetic Data)
--------------------------------------------------------------------
Generates synthetic multi-channel EEG windows for two classes:
- 0 = Calm
- 1 = Stress

Workflow
- Simulate EEG-like signals (alpha/beta differences) for each window
- Extract band-power features via Welch periodogram
- Train/test split, cross-val, RandomForest classifier
- Save metrics, confusion matrix, ROC-AUC plots, and feature importances
- Export the feature dataset to CSV

Run
    python eeg_stress_classification.py --samples 600 --channels 4 --fs 128 --duration 2.0

Outputs (created under ./outputs/)
- dataset.csv
- confusion_matrix.png
- roc_curve.png
- feature_importance.png
- report.txt

Author: Your Name
License: MIT
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------------- Signal Utilities ----------------------
def band_lims():
    # (low, high) in Hz
    return {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45)
    }

def bandpower_from_psd(freqs, psd, fmin, fmax):
    mask = (freqs >= fmin) & (freqs < fmax)
    return np.trapz(psd[mask], freqs[mask])

def simulate_eeg_window(fs, duration, channels, label, rng):
    """
    Create multi-channel EEG-like window.
    Calm: stronger alpha, moderate theta, lower beta.
    Stress: reduced alpha, elevated beta, slight gamma increase.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Base pink-ish noise (1/f)
    def pink_noise(n, rng):
        # Voss-McCartney style approx using cumulative sums of white noise at powers of two
        num_layers = 16
        white = rng.standard_normal((num_layers, n))
        cum = np.cumsum(white, axis=1)
        weights = 2.0 ** (-np.arange(num_layers))
        pink = (weights[:, None] * cum).sum(axis=0)
        pink = pink / np.std(pink)
        return pink

    X = np.zeros((channels, n))
    for ch in range(channels):
        sig = pink_noise(n, rng) * 0.3

        # Add oscillatory components
        # Alpha component stronger for calm, weaker for stress
        alpha_amp = rng.uniform(1.2, 1.8) if label == 0 else rng.uniform(0.4, 0.9)
        alpha_freq = rng.uniform(9, 12)
        sig += alpha_amp * np.sin(2*np.pi*alpha_freq*t + rng.uniform(0, 2*np.pi))

        # Beta component stronger for stress
        beta_amp = rng.uniform(0.5, 1.0) if label == 0 else rng.uniform(1.2, 2.0)
        beta_freq = rng.uniform(15, 22)
        sig += beta_amp * np.sin(2*np.pi*beta_freq*t + rng.uniform(0, 2*np.pi))

        # Theta moderate for calm, similar for stress
        theta_amp = rng.uniform(0.6, 1.0) if label == 0 else rng.uniform(0.5, 0.9)
        theta_freq = rng.uniform(5, 7.5)
        sig += theta_amp * np.sin(2*np.pi*theta_freq*t + rng.uniform(0, 2*np.pi))

        # Small gamma spurts for stress
        if label == 1:
            gamma_amp = rng.uniform(0.2, 0.6)
            gamma_freq = rng.uniform(32, 40)
            sig += gamma_amp * np.sin(2*np.pi*gamma_freq*t + rng.uniform(0, 2*np.pi))

        # Mild bandpass 0.5–45 Hz to mimic EEG preprocessing
        b, a = butter(4, [0.5/(fs/2), 45/(fs/2)], btype='band')
        sig = filtfilt(b, a, sig)

        # Normalize per channel
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        X[ch, :] = sig

    return X  # shape (channels, n)

def extract_bandpowers(window, fs):
    """Compute Welch PSD for each channel and integrate band powers."""
    bands = band_lims()
    ch, n = window.shape
    feats = {}
    for c in range(ch):
        freqs, psd = welch(window[c, :], fs=fs, nperseg=min(256, n))
        for bname, (lo, hi) in bands.items():
            feats[f"ch{c+1}_{bname}"] = bandpower_from_psd(freqs, psd, lo, hi)
        # Ratios often used as stress markers
        feats[f"ch{c+1}_beta_alpha_ratio"] = feats[f"ch{c+1}_beta"] / (feats[f"ch{c+1}_alpha"] + 1e-8)
        feats[f"ch{c+1}_theta_alpha_ratio"] = feats[f"ch{c+1}_theta"] / (feats[f"ch{c+1}_alpha"] + 1e-8)
    return feats

# ---------------------- Main Pipeline ----------------------
def generate_dataset(n_samples=600, channels=4, fs=128, duration=2.0, seed=42):
    rng = np.random.default_rng(seed)
    X_feats = []
    y = []
    for i in range(n_samples):
        label = 0 if i < (n_samples // 2) else 1  # balance
        window = simulate_eeg_window(fs, duration, channels, label, rng)
        feats = extract_bandpowers(window, fs)
        X_feats.append(feats)
        y.append(label)
    df = pd.DataFrame(X_feats)
    df["label"] = y
    return df

def train_and_evaluate(df, outdir: Path, seed=42):
    outdir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=seed
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=-1
    )
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Reports
    report = classification_report(y_test, y_pred, target_names=["Calm", "Stress"])
    cm = confusion_matrix(y_test, y_pred)

    # Save report
    report_txt = outdir / "report.txt"
    with report_txt.open("w") as f:
        f.write("EEG Stress Classification (Synthetic)\n")
        f.write("="*40 + "\n")
        f.write(f"CV Accuracy (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\n")
        f.write("Classification Report (Test):\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    # Confusion Matrix plot (single figure)
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xticks([0,1], ["Calm", "Stress"])
    plt.yticks([0,1], ["Calm", "Stress"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve (single figure)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    # Feature importances (single figure)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20
    labels = [f for f in df.drop(columns=["label"]).columns[indices]]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), labels, rotation=90)
    plt.ylabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance.png", dpi=200)
    plt.close()

    # Save scaled dataset to CSV
    cols = list(df.drop(columns=["label"]).columns) + ["label"]
    df_scaled = pd.DataFrame(np.column_stack([X_scaled, y]), columns=cols)
    df_scaled.to_csv(outdir / "dataset.csv", index=False)

    # Save metadata
    meta = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "roc_auc": float(roc_auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1])
    }
    with (outdir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "report_path": str(report_txt),
        "confusion_matrix": str(outdir / "confusion_matrix.png"),
        "roc_curve": str(outdir / "roc_curve.png"),
        "feature_importance": str(outdir / "feature_importance.png"),
        "dataset_csv": str(outdir / "dataset.csv"),
        "metadata_json": str(outdir / "metadata.json")
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=600, help="Total number of windows (>= 100)")
    ap.add_argument("--channels", type=int, default=4, help="Number of EEG channels")
    ap.add_argument("--fs", type=float, default=128.0, help="Sampling frequency (Hz)")
    ap.add_argument("--duration", type=float, default=2.0, help="Window duration (seconds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.samples < 100:
        raise SystemExit("--samples must be >= 100")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic EEG dataset...")
    df = generate_dataset(
        n_samples=args.samples,
        channels=args.channels,
        fs=args.fs,
        duration=args.duration,
        seed=args.seed
    )

    print("Training classifier and creating outputs...")
    paths = train_and_evaluate(df, outdir, seed=args.seed)

    print("Done. Outputs:")
    for k, v in paths.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
