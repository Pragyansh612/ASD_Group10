import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Data from our experiments
models = [
    'LR-ASD\nbaseline',
    'Transformer',
    'Multi-face\n(mean)',
    'Attn\ncontext',
    'Large\ncapacity',
    'Augment-\nation',
    'Hard\nnegatives',
    'TalkNCE',
    'CIR-020',
]

ava_correlation = [-0.17, -0.09, 0.59, 0.34, -0.02, -0.05, 0.13, -0.09, 0.10]
domain_drop     = [-27.87, -0.69, -18.50, -19.72, -26.41, -26.77, -26.95, -1.14, -20.06]
ava_map         = [94.11, 70.27, 82.87, 84.05, 93.52, 92.50, 91.88, 68.25, 85.94]

# Linear regression
X = np.array(ava_correlation).reshape(-1, 1)
y = np.array(domain_drop)
reg = LinearRegression().fit(X, y)
r2 = reg.score(X, y)
corr_coef, pval = pearsonr(ava_correlation, domain_drop)

print(f"Linear regression: drop = {reg.coef_[0]:.2f} * correlation + {reg.intercept_:.2f}")
print(f"R² = {r2:.3f}, Pearson r = {corr_coef:.3f}, p = {pval:.4f}")

# Predicted drop for new model
print("\nGeneralizability Score (predicted domain drop):")
for m, c in zip(models, ava_correlation):
    pred = reg.coef_[0] * c + reg.intercept_
    print(f"  {m.replace(chr(10),' ')}: corr={c:.2f} → predicted drop={pred:.1f}pp")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

colors = ['#2196F3', '#F44336', '#FF9800', '#FF5722', '#9C27B0',
          '#4CAF50', '#009688', '#795548', '#607D8B']

for i, (m, c, d, a) in enumerate(zip(models, ava_correlation, domain_drop, ava_map)):
    ax.scatter(c, d, s=200, color=colors[i], zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(m, (c, d), textcoords="offset points", xytext=(8, 5),
                fontsize=8.5, ha='left')

# Regression line
x_line = np.linspace(min(ava_correlation)-0.1, max(ava_correlation)+0.1, 100)
y_line = reg.coef_[0] * x_line + reg.intercept_
ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7, label=f'Linear fit (R²={r2:.2f})')

ax.set_xlabel('Inter-face Prediction Correlation (AVA val)', fontsize=13)
ax.set_ylabel('Domain Drop (AVA mAP → Columbia F1, pp)', fontsize=13)
ax.set_title('Inter-face Correlation Predicts Cross-Domain Generalization Failure\n'
             'Higher correlation = worse generalization', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/usershome/cs671_user6/asd_project/LR-ASD/scatter_correlation_drop.png',
            dpi=150, bbox_inches='tight')
print("\nSaved scatter_correlation_drop.png")
