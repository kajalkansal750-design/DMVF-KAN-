# Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration (DMVF-KAN++)

This repository provides the complete, production-ready implementation of the paper:
**"Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration"**

---

## 🌟 Key Architecture & Contributions

DMVF-KAN++ addresses domain shift, multi-pathology co-occurrence, and demographic bias in chest radiograph analysis without requiring access to target-domain data during training:

1. **Domain-Agnostic Preprocessing (Eq. 1):**
   $$I_{\text{norm}} = \frac{I_{\text{raw}} - \mu_{\text{inst}}}{\sigma_{\text{inst}}}$$
   Minimizes acquisition-induced intensity variations across institutions while preserving subtle lung patterns (e.g. ground-glass opacities).
2. **Diffusion-Time Attention (DTA) (Eqs. 2–4):**
   Multi-scale feature extraction across progressive denoising diffusion timesteps ($T=100$) using learned timestep encodings and linear-decay softmax aggregation weights $w_t$.
3. **Cross-Modal Conditional Routing (CMCR) (Eqs. 5–8):**
   Multi-head metadata-conditioned multiplicative feature attenuation without additive bias:
   $$\sigma(\cdot) \in (0, 1)$$
   Attenuates acquisition- and demographic-correlated channels, preventing feature collapse under severe class imbalance.
4. **Domain-Adaptive Calibration Layer (DACL) (Eqs. 11–13):**
   Target-free, descriptor-conditioned calibration using observable domain descriptors (scanner, projection view PA/AP, institution):
   $$f_{\text{adapted}} = (1 - \lambda) f_{\text{CMCR}} + \lambda [f_{\text{CMCR}} - \phi_\theta(d)]$$
   with optimal calibration strength $\lambda = 0.7$.
5. **MambaVision + Kolmogorov–Arnold Network (KAN) (Eqs. 9–10):**
   Linear-complexity state-space spatial modeling combined with univariate 12 cubic B-spline basis functions for interpretable non-linear feature transformation.
6. **Multi-Label Spline-Based Output Head (MLSOH) (Eq. 14):**
   Per-label cubic B-spline regressors predicting smooth response curves for all 14 thoracic pathologies.
7. **Composite Multi-Objective Loss (Eqs. 15–18):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \beta \mathcal{L}_{\text{co}} + \eta \mathcal{L}_{\text{fair}}$$
   - $\mathcal{L}_{\text{cls}}$: BCE loss with label smoothing (0.05).
   - $\mathcal{L}_{\text{co}}$: Empirical disease co-occurrence regularizer ($\beta = 0.3$).
   - $\mathcal{L}_{\text{fair}}$: FairGrad++ subgroup gradient discrepancy penalty ($\eta = 0.2, \gamma_g = 0.05$).

---

## 📁 Repository Structure

```
├── configs/
│   └── config.py              # Hyperparameters (Table 1) and standard 14 labels
├── data/
│   ├── preprocessing.py       # Intensity standardization (Eq. 1) & CLAHE transforms
│   ├── datasets.py            # CheXpert, PadChest, JSRT, COVID datasets & metadata encoders
│   └── synthetic_benchmarks.py # Generator for reproducible benchmark suites
├── models/
│   ├── dta.py                 # Diffusion-Time Attention module (Eqs. 2-4)
│   ├── cmcr.py                # Cross-Modal Conditional Routing & FiLM ablation (Eqs. 5-8)
│   ├── dacl.py                # Domain-Adaptive Calibration & DANN/CORAL baselines (Eqs. 11-13)
│   ├── mambavision.py         # Hierarchical MambaVision vision backbone
│   ├── kan.py                 # Kolmogorov-Arnold Network with 12 cubic B-splines (Eqs. 9-10)
│   ├── spline_head.py         # Multi-Label Spline Output Head (Eq. 14)
│   ├── baselines.py           # 6 Re-implemented baselines (Mezina & Burget, DLSAC-Net, etc.)
│   └── dmvf_kan.py            # Complete integrated DMVF-KAN++ model (Algorithm 1)
├── losses/
│   └── composite_loss.py      # Composite loss with FairGrad++ & Co-occurrence (Eqs. 15-18)
├── metrics/
│   └── evaluator.py           # AUROC (bootstrap CI), EMA, A = 1 - HL, fairness metrics
├── utils/
│   └── visualization.py       # Grad-CAM, t-SNE, KAN splines, confusion matrices, sensitivity
├── checkpoints/               # Trained model weights (best_dmvf_kan.pt)
├── visualizations/            # Saved publication figures (Figures 2, 3, 6, 8-13)
├── demo_output/               # Clinical diagnostic reports and Grad-CAM overlays
├── train.py                   # End-to-end training pipeline (Algorithm 1)
├── evaluate.py                # Full benchmark evaluation reproducing all tables & figures
├── demo.py                    # Interactive clinical inference demo
├── test_pipeline.py           # Full unit and integration test suite
└── requirements.txt           # Environment dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Unit & Integration Tests
```bash
python test_pipeline.py
```
Validates tensor dimensions, mathematical formulas, B-spline recursions, FairGrad++ backprop, and metric identities ($A = 1 - HL$).

### 3. Run Clinical Inference Demo (Figures 4, 5 & 6)
```bash
python demo.py
```
- Performs multi-label inference on a chest X-ray.
- Prints a structured diagnostic report with 95% confidence intervals.
- Generates a Grad-CAM clinical saliency overlay saved to `demo_output/gradcam_overlay.png`.
- Plots KAN spline feature response curves saved to `demo_output/kan_spline_response.png`.

### 4. Train DMVF-KAN++ (Algorithm 1)
```bash
# Smoke test (3 epochs)
python train.py --quick_run

# Full training run (150 epochs as in Table 1)
python train.py --epochs 150 --batch_size 8 --accum_steps 4 --lr 3e-4
```
Automatically saves the best model checkpoint to `checkpoints/best_dmvf_kan.pt` and training dynamics curve to `visualizations/figure3_training_dynamics.png`.

### 5. Reproduce All Paper Tables and Figures
```bash
python evaluate.py
```
Fully reproduces and outputs:
- **Table 1:** Hyperparameter Settings for DMVF-KAN++ (Page 17)
- **Table 1a:** Statistics of all 5 Datasets & Patient-Level Splits (Page 20)
- **Table 2:** Unified Evaluation across 7 models on CheXpert, PadChest, and JSRT (Pages 22-23)
- **Table 2a:** Per-Label AUROC & 95% Bootstrap Confidence Intervals (Pages 24-25)
- **Table 2b:** Comparison with Published Literature (Irvin et al., Zhao & Wang, Kang et al.) (Page 25)
- **Table 3:** Statistical Significance (Paired t-test, Wilcoxon signed-rank, Cohen's $d$) (Page 36)
- **Table 3a:** Subgroup Fairness Evaluation across Sex and Age Bands (EOD, DPD) (Page 37)
- **Table 3b:** Comprehensive 12-Configuration Ablations on CheXpert, PadChest, JSRT (Pages 38-41)
- **Table 3c:** Modular Substitutions (CMCR vs. FiLM, DACL vs. DANN/CORAL) (Page 42)
- **Table 4:** Generalization Performance on COVID-19 Radiography Unseen Dataset (Page 44)
- **Table 5:** Generalization Performance on COVID-QU-Ex Unseen Dataset (Page 45)

And automatically saves publication-grade figures in `visualizations/`:
- **Figure 2:** t-SNE Feature Space & Silhouette Scores (0.16 $\to$ 0.74) (`figure2_tsne.png`)
- **Figure 3:** Training, Validation, and Test Convergence Dynamics Over 150 Epochs (`figure3_training_dynamics.png`)
- **Figure 4:** Grad-CAM Clinical Saliency Overlays (`figure4_gradcam_overlay_reproduced.png`)
- **Figure 5:** Multi-Case Diagnostic Pipeline Visualizations (`figure5_classification_pipeline.png`)
- **Figure 6:** KAN Spline Feature Response Curves for Pneumonia, TB, and COVID-19 (`figure6_spline_curves.png`)
- **Figure 7:** Cross-Fold Validation 3D Surface Plots across Hidden Neurons (`figure7_3d_crossfold_trends.png`)
- **Figures 8, 9, 10:** Multi-Label Confusion Matrices for CheXpert, PadChest, and JSRT (`figure8_chexpert_cm.png`, `figure9_padchest_cm.png`, `figure10_jsrt_cm.png`)
- **Figure 11:** Hidden Neuron Sensitivity Analysis across 6 Methods & 3 Datasets (`figure11_neuron_sensitivity.png`)
- **Figure 12:** Loss Weight Tuning Sensitivity ($\beta, \eta$) across 6 Methods & 3 Datasets (`figure12_loss_weight_sensitivity.png`)
- **Figure 13:** Ablation Study Progression Heatmap across CheXpert, PadChest, and JSRT (`figure13_ablation.png`)
