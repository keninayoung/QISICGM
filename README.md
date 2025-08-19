# Quantum-Inspired Stacked Integrated Concept Graph Model (QISICGM) for Diabetes Prediction

## Overview
The QISICGM is a novel machine learning framework designed to predict diabetes risk using the PIMA Indians Diabetes dataset, augmented with synthetic data. 
This project leverages quantum-inspired techniques, including a self-improving graph-based embedding model (`QISICGM`) and sequence generation inspired by quantum walks, combined with a hybrid ensemble approach. 
The meta model is trained on embeddings to achieve sub-second predictions while reducing false negatives, making it suitable for real-time clinical applications.

- **Author**: Kenneth Young, PhD (USF-HII)
- **License**: MIT
- **Date**: August 19, 2025

## Features
- Quantum-inspired embedding generation with self-improvement.
- Hybrid ensemble using Random Forest, Extra Trees, Transformer, FFNN, and CNN base models.
- Meta model trained on embeddings for fast predictions (<1s).
- Comprehensive visualization of performance metrics and confusion matrix.
- Reduction of false negatives from 426 to 107 on the test set.

## Requirements
Install the required Python packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

Training: Execute qisicgm_stacked.py to train the QISICGM, generate embeddings, and save models. The script creates models/ and plots/ directories.
Prediction: Run demo_predictions.py to predict diabetes risk for new patients. Example output includes probabilities and classifications (e.g., "High risk" or "Low risk") in under 1 second.
Plots: Check plots/ for performance visualizations (e.g., confusion matrix, ROC curves).

## Results

OOF Performance: F1 = 0.8881, AUC = 0.8376, Threshold = 0.4360
Confusion Matrix: TN = 462, FP = 357, FN = 107, TP = 1842
Demo Predictions (0.03s for 2 patients):
Patient 1: prob=0.6010 → High risk
Patient 2: prob=0.0406 → Low risk

## Plots
The following plots are saved in the plots/ directory:

- performance_table.png: Table of fold-wise metrics.
- summary_bars.png: Bar charts of F1, Precision, Recall, AUC.
- roc_curves_meta.png: ROC curves for each fold and OOF.
- pr_curves_meta.png: Precision-recall curves.
- calibration_oof_meta.png: Calibration curve and probability histogram.
- confusion_matrix.png: Heatmap and bars for TN, FP, FN, TP.
- score_distributions_oof_meta.png: Probability distributions by class.

## Contributing
1. Fork the repository.
2. Create a feature branch: git checkout -b feature-name.
3. Commit changes: git commit -m "Description".
4. Push to the branch: git push origin feature-name.
5. Submit a pull request.
