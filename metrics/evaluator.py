"""
Quantitative Evaluation and Subgroup Fairness Metrics
Implements Section 4.3, Section 4.3.5 (Table 3), and Section 4.3.6 (Table 3a) of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Strictly implements:
- Macro AUROC (mAUROC) & Per-Label AUROC with 95% bootstrap confidence intervals
- Exact-Match Accuracy (EMA)
- Label-Wise Accuracy (A) and Hamming Loss (HL) ensuring A = 1 - HL
- Macro Sensitivity (S), Precision (P), F1 (F), MCC, and Jaccard Index (JI)
- Subgroup Fairness Metrics: \Delta AUROC (sex & age), worst-group AUROC, EOD, DPD
- Statistical Significance: Paired t-test, Wilcoxon signed-rank test, Cohen's d
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


def compute_bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes AUROC and 95% bootstrap confidence interval (Table 2a).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5
        
    base_auroc = roc_auc_score(y_true, y_pred)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = roc_auc_score(y_true[idx], y_pred[idx])
        bootstrapped_scores.append(score)
        
    if len(bootstrapped_scores) == 0:
        return base_auroc, base_auroc, base_auroc
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrapped_scores, 100 * alpha))
    ci_upper = float(np.percentile(bootstrapped_scores, 100 * (1.0 - alpha)))
    return float(base_auroc), ci_lower, ci_upper


def evaluate_multilabel_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Computes all unified multi-label classification metrics (Section 4.3):
    - mAUROC: primary metric (macro-averaged over labels)
    - EMA: exact-match accuracy
    - A: label-wise accuracy (1 - HL)
    - HL: Hamming loss
    - S: macro Sensitivity / Recall (%)
    - P: macro Precision (%)
    - F: macro F1-score (%)
    - MCC: macro Matthews Correlation Coefficient
    - JI: sample-averaged Jaccard Index (%)
    """
    n_samples, n_classes = y_true.shape
    if thresholds is None:
        thresholds = np.full(n_classes, 0.5)
        
    # Binary predictions y_hat = 1[p >= tau]
    y_bin = (y_probs >= thresholds).astype(int)
    
    # 1. AUROC (per-label & macro-average)
    per_label_auroc = []
    per_label_ci = []
    for k in range(n_classes):
        if len(np.unique(y_true[:, k])) > 1:
            score, ci_low, ci_high = compute_bootstrap_auroc_ci(y_true[:, k], y_probs[:, k])
        else:
            score, ci_low, ci_high = 0.5, 0.5, 0.5
        per_label_auroc.append(score)
        per_label_ci.append((ci_low, ci_high))
        
    mAUROC = float(np.mean(per_label_auroc))
    
    # 2. Exact-Match Accuracy (EMA)
    # Fraction of images where full label vector is identical
    ema = float(np.mean(np.all(y_bin == y_true, axis=1))) * 100.0
    
    # 3. Label-Wise Accuracy (A) and Hamming Loss (HL)
    hl = float(hamming_loss(y_true, y_bin))
    label_acc = (1.0 - hl) * 100.0  # Mathematical identity A = 1 - HL
    
    # 4. Per-label Sensitivity, Precision, F1, MCC
    sens_list, prec_list, f1_list, mcc_list = [], [], [], []
    for k in range(n_classes):
        yt, yp = y_true[:, k], y_bin[:, k]
        s = recall_score(yt, yp, zero_division=0)
        p = precision_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        mcc = matthews_corrcoef(yt, yp)
        
        sens_list.append(s)
        prec_list.append(p)
        f1_list.append(f)
        mcc_list.append(mcc)
        
    macro_s = float(np.mean(sens_list)) * 100.0
    macro_p = float(np.mean(prec_list)) * 100.0
    macro_f1 = float(np.mean(f1_list)) * 100.0
    macro_mcc = float(np.mean(mcc_list))
    
    # 5. Jaccard Index (JI): sample-averaged |y_hat \cap y| / |y_hat \cup y|
    jaccards = []
    for i in range(n_samples):
        intersection = np.sum((y_bin[i] == 1) & (y_true[i] == 1))
        union = np.sum((y_bin[i] == 1) | (y_true[i] == 1))
        jaccards.append(1.0 if union == 0 else (intersection / union))
    ji = float(np.mean(jaccards)) * 100.0
    
    results = {
        "mAUROC": mAUROC,
        "EMA": ema,
        "Accuracy": label_acc,
        "HammingLoss": hl,
        "Sensitivity": macro_s,
        "Precision": macro_p,
        "F1": macro_f1,
        "MCC": macro_mcc,
        "JaccardIndex": ji,
        "per_label_auroc": per_label_auroc,
        "per_label_ci": per_label_ci,
        "class_names": class_names
    }
    return results


def evaluate_fairness_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    subgroups: np.ndarray,
    sex_indices: Optional[List[int]] = None,
    age_band_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Subgroup Fairness Evaluation (Section 4.3.6, Table 3a):
    - \Delta AUROC (sex): between male and female
    - \Delta AUROC (age): between age bands
    - Worst-group mAUROC
    - Equalized-Odds Difference (EOD)
    - Demographic-Parity Difference (DPD)
    """
    unique_groups = np.unique(subgroups)
    group_aurocs = {}
    
    for g in unique_groups:
        mask = (subgroups == g)
        if np.sum(mask) >= 10:  # Minimum samples for reliable AUROC
            yt_g, yp_g = y_true[mask], y_probs[mask]
            # Compute macro AUROC for this group
            aurocs = []
            for k in range(yt_g.shape[1]):
                if len(np.unique(yt_g[:, k])) > 1:
                    aurocs.append(roc_auc_score(yt_g[:, k], yp_g[:, k]))
            if aurocs:
                group_aurocs[g] = float(np.mean(aurocs))
                
    worst_group_auroc = min(group_aurocs.values()) if group_aurocs else 0.0
    
    # Sex-wise \Delta AUROC (Male: subgroups 0..3, Female: subgroups 4..7)
    male_mask = (subgroups < 4)
    fem_mask = (subgroups >= 4)
    
    m_aurocs, f_aurocs = [], []
    for k in range(y_true.shape[1]):
        if np.sum(male_mask) > 5 and len(np.unique(y_true[male_mask, k])) > 1:
            m_aurocs.append(roc_auc_score(y_true[male_mask, k], y_probs[male_mask, k]))
        if np.sum(fem_mask) > 5 and len(np.unique(y_true[fem_mask, k])) > 1:
            f_aurocs.append(roc_auc_score(y_true[fem_mask, k], y_probs[fem_mask, k]))
            
    m_mean = float(np.mean(m_aurocs)) if m_aurocs else 0.0
    f_mean = float(np.mean(f_aurocs)) if f_aurocs else 0.0
    delta_auroc_sex = abs(m_mean - f_mean)
    
    # Age-wise \Delta AUROC (across 4 age bands)
    age_aurocs = []
    for band in range(4):
        band_mask = ((subgroups % 4) == band)
        b_scores = []
        for k in range(y_true.shape[1]):
            if np.sum(band_mask) > 5 and len(np.unique(y_true[band_mask, k])) > 1:
                b_scores.append(roc_auc_score(y_true[band_mask, k], y_probs[band_mask, k]))
        if b_scores:
            age_aurocs.append(float(np.mean(b_scores)))
            
    delta_auroc_age = (max(age_aurocs) - min(age_aurocs)) if len(age_aurocs) > 1 else 0.0
    
    # Equalized Odds Difference (EOD) & Demographic Parity Difference (DPD)
    y_bin = (y_probs >= 0.5).astype(int)
    
    # TPR, FPR, and Positive Rate per group
    tprs, fprs, prs = [], [], []
    for g in [male_mask, fem_mask]:
        yt_sub = y_true[g]
        yp_sub = y_bin[g]
        
        # TPR = TP / (TP + FN)
        tpr = np.mean([recall_score(yt_sub[:, k], yp_sub[:, k], zero_division=0) for k in range(yt_sub.shape[1])])
        
        # FPR = FP / (FP + TN)
        fpr_list = []
        for k in range(yt_sub.shape[1]):
            tn = np.sum((yt_sub[:, k] == 0) & (yp_sub[:, k] == 0))
            fp = np.sum((yt_sub[:, k] == 0) & (yp_sub[:, k] == 1))
            fpr_list.append(fp / (fp + tn + 1e-7))
        fpr = np.mean(fpr_list)
        
        # Positive prediction rate
        pr = np.mean(yp_sub)
        
        tprs.append(tpr)
        fprs.append(fpr)
        prs.append(pr)
        
    eod = max(abs(tprs[0] - tprs[1]), abs(fprs[0] - fprs[1]))
    dpd = abs(prs[0] - prs[1])
    
    return {
        "mAUROC_Female": f_mean,
        "mAUROC_Male": m_mean,
        "delta_AUROC_sex": delta_auroc_sex,
        "worst_group_AUROC": worst_group_auroc,
        "delta_AUROC_age": delta_auroc_age,
        "EOD": eod,
        "DPD": dpd
    }


def compute_statistical_significance(
    scores_proposed: List[float],
    scores_baseline: List[float]
) -> Dict[str, float]:
    """
    Parametric paired t-test, Wilcoxon signed-rank test, and Cohen's d effect size (Section 4.3.5, Table 3).
    """
    p = np.array(scores_proposed)
    b = np.array(scores_baseline)
    diff = p - b
    
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    
    # 95% Confidence Interval
    ci = stats.t.interval(0.95, df=len(diff)-1, loc=mean_diff, scale=stats.sem(diff)) if len(diff) > 1 else (mean_diff, mean_diff)
    
    # Cohen's d
    cohen_d = mean_diff / (std_diff + 1e-7)
    
    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(p, b) if len(diff) > 1 else (0.0, 1.0)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(p, b)
    except Exception:
        p_wilcoxon = p_ttest
        
    return {
        "mean_diff": mean_diff,
        "std_dev": std_diff,
        "ci_95": (float(ci[0]), float(ci[1])),
        "cohens_d": float(cohen_d),
        "p_value_ttest": float(p_ttest),
        "p_value_wilcoxon": float(p_wilcoxon)
    }
