from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def mean_roc_curve(df, label_col='flood', score_col='RASTERVALU'):

    X = df[score_col].values
    y = df[label_col].values

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    mean_fpr = np.linspace(0, 1, 500)

    tprs = []
    aucs = []

    for train_idx, test_idx in cv.split(X, y):

        y_test = y[test_idx]
        score_test = X[test_idx]

        fpr, tpr, _ = roc_curve(y_test, score_test)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0

        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1

    mean_auc = np.mean(aucs)

    return mean_fpr, mean_tpr, mean_auc

# Calculate mean ROC curves
fgm_fpr, fgm_tpr, fgm_auc = mean_roc_curve(df_fgm_filtered)
few_fpr, few_tpr, few_auc = mean_roc_curve(df_few_filtered)
ahp_fpr, ahp_tpr, ahp_auc = mean_roc_curve(df_ahp_filtered)

# Plot
plt.figure(figsize=(10,8))

plt.plot(
    fgm_fpr,
    fgm_tpr,
    linewidth=2.5,
    label=f'Fuzzy Geometric Mean (AUC = {fgm_auc:.3f})'
)

plt.plot(
    few_fpr,
    few_tpr,
    linewidth=2.5,
    label=f"Chang's Extent Analysis (AUC = {few_auc:.3f})"
)

plt.plot(
    ahp_fpr,
    ahp_tpr,
    linewidth=2.5,
    label=f'AHP (AUC = {ahp_auc:.3f})'
)

plt.plot(
    [0,1],
    [0,1],
    'k--',
    linewidth=1.5,
    label='Random Classifier'
)

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)

plt.title(
    'ROC-AUC',
    fontsize=14
)

plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    'Mean_ROC_10Fold.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()

# Summary table
summary_df = pd.DataFrame({
    'Method': [
        'Fuzzy Geometric Mean',
        "Chang's Extent Analysis",
        'AHP'
    ],
    'Mean AUC': [
        fgm_auc,
        few_auc,
        ahp_auc
    ]
})

print(summary_df.round(4))
summary_df.to_csv('Mean_AUC_10Fold.csv', index=False)
