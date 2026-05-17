import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import umap


def run_umap_pipeline(X_scaled, y, N, cmap_seq, f, output_dir):
    """Thực thi độc lập thuật toán UMAP và kết xuất đồ họa liên kết không gian phi tuyến."""
    f.write("\n" + "=" * 60 + "\nUMAP ANALYSIS\n" + "=" * 60 + "\n")

    m2_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled.values)
    m2_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(X_scaled.values)
    lbl = "UMAP"

    m2_df = pd.DataFrame(m2_2d, columns=[f'{lbl}1', f'{lbl}2'])
    f.write(f"\nThống kê {lbl} components:\n")
    f.write(m2_df.describe().round(4).to_string() + "\n")

    f.write(f"\nTương quan {lbl} với charges:\n")
    for i, col in enumerate(m2_df.columns):
        r = np.corrcoef(m2_df.iloc[:, i], y.values)[0, 1]
        f.write(f"  {col} vs charges: r = {r:+.4f}\n")

    q33 = np.percentile(y.values, 33)
    q66 = np.percentile(y.values, 66)
    groups = np.where(y.values < q33, 0, np.where(y.values < q66, 1, 2))
    g_lbls = ['Thấp', 'Trung bình', 'Cao']
    g_cols = ['#2E86AB', '#F18F01', '#E84855']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{lbl} — Trực quan hóa phi tuyến", fontsize=13, fontweight='bold')

    ax = axes[0]
    sc = ax.scatter(m2_2d[:, 0], m2_2d[:, 1], c=y.values, cmap=cmap_seq, alpha=0.7, s=20, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='charges')
    ax.set_xlabel(f"{lbl}1")
    ax.set_ylabel(f"{lbl}2")
    ax.set_title(f"{lbl} 2D — tô màu theo charges")

    ax = axes[1]
    for g, (gl, gc) in enumerate(zip(g_lbls, g_cols)):
        mask = groups == g
        ax.scatter(m2_2d[mask, 0], m2_2d[mask, 1], label=f'{gl} (n={mask.sum()})', color=gc, alpha=0.65, s=20,
                   edgecolors='none')
    ax.set_xlabel(f"{lbl}1")
    ax.set_ylabel(f"{lbl}2")
    ax.set_title(f"{lbl} 2D — 3 nhóm charges")
    ax.legend(fontsize=8)

    ax = axes[2]
    if N > 5:
        kde = gaussian_kde(m2_2d.T)
        dens = kde(m2_2d.T)
        sc2 = ax.scatter(m2_2d[:, 0], m2_2d[:, 1], c=dens, cmap='hot_r', alpha=0.7, s=20, edgecolors='none')
        plt.colorbar(sc2, ax=ax, label='Mật độ')
    else:
        ax.scatter(m2_2d[:, 0], m2_2d[:, 1], color='#2E86AB', s=40)
    ax.set_xlabel(f"{lbl}1")
    ax.set_ylabel(f"{lbl}2")
    ax.set_title(f"{lbl} — Mật độ KDE")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_nonlinear_diagnostics.png", dpi=200)
    plt.close()

    pair_data = m2_3d[:, :3]
    pair_labels = [f'{lbl}{i + 1}' for i in range(3)]
    pairs_m2 = list(itertools.combinations(range(3), 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{lbl} — Scatter Plot cặp thành phần (3 components)", fontsize=13, fontweight='bold')
    for idx, (i, j) in enumerate(pairs_m2):
        ax = axes[idx]
        sc2 = ax.scatter(pair_data[:, i], pair_data[:, j], c=y.values, cmap=cmap_seq, alpha=0.65, s=15,
                         edgecolors='none')
        plt.colorbar(sc2, ax=ax, shrink=0.85)
        ax.set_xlabel(pair_labels[i], fontsize=9)
        ax.set_ylabel(pair_labels[j], fontsize=9)
        ax.set_title(f"{pair_labels[i]} vs {pair_labels[j]}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_3components_pairs.png", dpi=200)
    plt.close()

    f.write("\n[Giải thích]\n")
    f.write("- UMAP không có explained variance\n")
    f.write("- Dùng để biểu diễn cấu trúc phi tuyến\n")
    f.write("- PCA giữ ít thông tin → dữ liệu có thể phi tuyến\n")
    f.write("- UMAP giúp biểu diễn cấu trúc phi tuyến tốt hơn\n")
    f.write("- UMAP phù hợp hơn PCA để trực quan dataset này\n")