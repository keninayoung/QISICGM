# Quantum-Inspired Stacked Integrated Concept Graph Model (QISICGM) for Diabetes Prediction
Author: Kenneth Young, PhD

## Overview
The Quantum-Inspired Stacked Integrated Concept Graph Model (QISICGM) is an innovative machine learning framework designed to predict diabetes risk using the PIMA Indians Diabetes dataset, augmented with synthetic data to improve balance and diversity. The model processes patient features (e.g., glucose, BMI, age) to output a probability of diabetes, with a focus on reducing false negatives for early detection in clinical settings. It combines quantum-inspired embedding generation, graph-based learning, and a hybrid stacked ensemble to achieve high performance (OOF F1 score of 0.8881, AUC of 0.8376) while enabling sub-second predictions on new patients.

## Key Components and Workflow

Data Preparation: The PIMA dataset (768 samples, 8 features) is augmented with 2000 synthetic samples (total 2768 samples, 1949 positives) to address class imbalance. Features are imputed, engineered (e.g., Glucose_BMI = glucose * BMI), and scaled.
Quantum-Inspired Embedding Generation: A core module (QISICGM) creates embeddings from input features using a self-improving graph structure. It initializes a k-nearest neighbor graph and refines embeddings through gradient-based optimization, mimicking quantum processes. Embeddings are used for base model training and meta learning.

Hybrid Ensemble:

Base Models (Trained once on the full dataset's embeddings):
  - Random Forest (RF) and Extra Trees (ET) for tree-based diversity.
  - Transformer for sequence processing (attention inspired by quantum entanglement).
  - Feedforward Neural Network (FFNN) and Convolutional Neural Network (CNN) for neural-based learning.

These models influence the embedding space during QISICGM's self-improvement, adding quantum-inspired diversity.

Meta-Learner: 
A LogisticRegression model trained on embeddings (128-dimensional) from the full dataset, calibrated with 5-fold cross-validation to optimize the threshold (0.4360) for balanced recall/precision.
Prediction Pipeline: For new patients, preprocess data, generate embeddings with qm_final, and predict with the meta model in <1 second on CPU.
Visualization: Generates 7 plots (e.g., performance table, ROC/PR curves, confusion matrix, score distributions) in "plots/" for analysis.
Results: OOF F1 = 0.8881, AUC = 0.8376, confusion matrix: TN=462, FP=357, FN=107, TP=1842 (FN reduced from 426 to 107).

The pipeline is implemented in qisicgm_stacked.py (training) and demo_predictions.py (inference), with a requirements.txt for dependencies.

## How It Is Quantum-Inspired
The QISICGM draws inspiration from quantum mechanics to enhance classical ML, particularly in embedding and graph processing:

Phase Feature Map: Transforms input features into a higher-dimensional space using cosine and sine functions (cos(alpha * x), sin(alpha * x)), mimicking quantum amplitude encoding where features are mapped to quantum states with phase information. This creates a non-linear representation, akin to quantum superposition.

Self-Improving Graph Structure: Builds a k-nearest neighbor graph from embeddings and refines it with gradient-based optimization and pruning, inspired by quantum annealing (e.g., D-Wave systems) where systems evolve to low-energy states. This simulates quantum optimization for feature relationships.

Quantum Walk Simulation: In sequence generation (build_sequences_from_graph_with_mask), a simulated quantum walk computes neighbor probabilities with interference terms (walk_probs = walk_probs * (1 - walk_probs)), reflecting quantum superposition and interference. This generates sequences for models like Transformer and CNN, adding quantum-inspired randomness and exploration.

Hybrid Ensemble Integration: The Transformer’s attention mechanism mimics quantum entanglement (correlating distant features), while CNN convolutions parallel quantum circuit gates. Base models trained on these embeddings indirectly incorporate quantum-inspired diversity.

These elements make QISICGM "quantum-inspired," meaning it uses classical analogs of quantum concepts for improved performance, without actual quantum hardware, enabling scalability on standard CPUs.

## Novelty of the Approach
While quantum-inspired ML (QML) techniques exist (e.g., quantum kernels in Schuld & Killoran, 2019, or graph quantum walks in Biamonte et al., 2017), QISICGM’s novelty lies in its integrated hybrid framework for imbalanced medical data:

Integrated Embedding and Stacking: Unlike traditional QML focusing on quantum circuits or kernels, QISICGM combines a self-improving graph embedding (quantum annealing-inspired) with sequence generation (quantum walk-inspired) to feed a hybrid ensemble (tree-based + neural models). This "stacked integrated concept graph" is unique, blending quantum optimization with classical ML for diabetes prediction.

Hybrid Ensemble with Embeddings: The one-time training of base models to enrich embeddings, saved for reuse, is novel for efficiency in QML hybrids. It reduces runtime while maintaining diversity, contrasting pure quantum models (e.g., Havlíček et al., 2019) that require hardware.

Sub-Second Predictions on Imbalanced Data: Achieves FN reduction (107 from 426) with a LogisticRegression meta model on 128-dimensional embeddings, enabling real-time use, a rarity in QML for healthcare (most focus on accuracy, not speed).
Practical Novelty: Applied to PIMA dataset with synthetic augmentation, yielding OOF F1 0.8881 and AUC 0.8376, outperforming baselines while addressing QML scalability challenges.

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
Confusion Matrix: TN = 462, FP = 357, FN = 107, TP = 1842
<img width="1783" height="732" alt="confusion_oof_meta_bars" src="https://github.com/user-attachments/assets/daa3d3b6-5cb7-49cf-b3c8-975c5f29d743" />

Concept Graph:
<img width="1530" height="1261" alt="concept_graph_fold1" src="https://github.com/user-attachments/assets/552ea8cc-2b34-460a-9ca5-2f687fbea555" />

OOF Performance: F1 = 0.8881, AUC = 0.8376, Threshold = 0.4360
<img width="1425" height="985" alt="performance_table" src="https://github.com/user-attachments/assets/974dc315-0212-46b3-b5ac-cb7aaa7d0e66" />

AUC/ROC Curves:
<img width="1268" height="932" alt="roc_curves_meta" src="https://github.com/user-attachments/assets/ec39dfe7-2a55-4054-aab6-c3eedfdaa59c" />

Score Distributions Out-Of-Fold (OOF) Meta:
<img width="1268" height="789" alt="score_distributions_oof_meta" src="https://github.com/user-attachments/assets/89216afb-221e-42a3-afdc-931196b58986" />

Summary Metrics Across Folds:
<img width="1268" height="789" alt="summary_bars" src="https://github.com/user-attachments/assets/dd7ca04f-bba3-4235-865f-9a25ee7c3720" />

Precision-Recall Curves:
<img width="1281" height="932" alt="pr_curves_meta" src="https://github.com/user-attachments/assets/e7b78ab2-e42b-4a35-a886-b045c2d0a305" />

Calibration OOF Meta:
<img width="1484" height="731" alt="calibration_oof_meta" src="https://github.com/user-attachments/assets/71e0ec5c-a7c0-4876-ae49-04619bcea19f" />

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

## License: MIT

## Contributing
  1. Fork the repository.
  2. Create a feature branch: git checkout -b feature-name.
  3. Commit changes: git commit -m "Description".
  4. Push to the branch: git push origin feature-name.
  5. Submit a pull request.
