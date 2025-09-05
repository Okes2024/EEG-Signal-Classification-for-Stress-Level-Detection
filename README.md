EEG-Signal-Classification-for-Stress-Level-Detection

🧠 EEG Signal Classification using synthetic data to detect stress levels. This project simulates multi-channel EEG signals, extracts frequency-band features, and trains a machine learning model to classify mental states as Calm or Stress.

🚀 Project Overview

Stress detection from EEG signals is a growing field in neurotechnology and mental health monitoring.
Since real EEG datasets can be difficult to obtain, this project uses synthetic EEG data to:

Generate >100 EEG windows with realistic oscillatory patterns.

Extract band power features (delta, theta, alpha, beta, gamma).

Compute stress markers (e.g., beta/alpha ratio).

Train a Random Forest classifier with cross-validation.

Evaluate performance using confusion matrix, ROC curve, and feature importance plots.

📂 Repository Structure
├── eeg_stress_classification.py   # Main script
├── outputs/                       # Generated results
│   ├── dataset.csv                # Feature dataset
│   ├── report.txt                 # Classification report
│   ├── confusion_matrix.png       # Confusion matrix (test set)
│   ├── roc_curve.png              # ROC curve (test set)
│   ├── feature_importance.png     # Top feature importance plot
│   └── metadata.json              # Performance metadata
└── README.md                      # Project documentation

🔑 Features

Synthetic EEG generation (calm vs stress).

Band power extraction via Welch periodogram.

Ratios for stress biomarkers (β/α, θ/α).

Balanced dataset (>100 samples).

Train/test split with 5-fold cross-validation.

Export of dataset and evaluation plots.

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/EEG-Signal-Classification-for-Stress-Level-Detection.git
cd EEG-Signal-Classification-for-Stress-Level-Detection
pip install numpy scipy pandas scikit-learn matplotlib

🧑‍💻 Usage

Run the main script with default parameters (600 samples, 4 channels, 128 Hz, 2s windows):

python eeg_stress_classification.py


Customize parameters:

python eeg_stress_classification.py --samples 1000 --channels 8 --fs 256 --duration 3.0 --outdir results

📊 Results

Generated outputs include:

dataset.csv → feature dataset with labels.

report.txt → accuracy, precision, recall, F1-score.

Confusion Matrix → classification performance.

ROC Curve → AUC evaluation.

Feature Importances → top EEG band features for stress detection.

🌍 Applications

Cognitive load monitoring.

Stress and fatigue detection.

Brain-computer interfaces (BCI).

Prototyping mental health monitoring tools.

🤝 Contributing

Contributions are welcome! Please fork this repo, create a feature branch, and submit a pull request.

📜 License

This project is licensed under the MIT License. See LICENSE
 for details.

📧 Contact

Maintainer: Imoni Okes
Github: https://github.com/Okes2024

🔗 LinkedIn
