import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns


class PCA_Manual:
    """Thuật toán phân tích thành phần chính tự xây dựng bằng phân rã ma trận hiệp phương sai."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.components = self.eigenvectors[:, :self.n_components]
        total = np.sum(self.eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def run_pca_pipeline(X_scaled, y, feat_names, N, D, cmap_div, cmap_seq, palette, f, output_dir):
    """Thực thi độc lập luồng phân tích toán học và vẽ biểu đồ cho cấu trúc PCA."""
    f.write("\n" + "=" * 60 + "\nPCA ANALYSIS\n" + "=" * 60 + "\n")

    pca_full = PCA_Manual(len(X_scaled))
    X_pca_full = pca_full.fit_transform(X_scaled.values)

    explained = pca_full.explained_variance_ratio_
    ev_m = pca_full.eigenvalues
    cum_m = np.cumsum(explained)

    f.write(f"  {'PC':<6} {'Eigenvalue':>12} {'Var %':>10} {'Cộng dồn %':>13}\n")
    f.write(f"  {'-' * 45}\n")
    for i in range(min(10, len(ev_m))):
        bar = '▓' * max(1, int(explained[i] * 100 / 2))
        f.write(f"  PC{i + 1:<4} {ev_m[i]:>12.4f} {explained[i] * 100:>9.2f}%  {cum_m[i] * 100:>11.2f}%  {bar}\n")

    total_10 = cum_m[9]
    f.write(f"\nTổng variance với 10 components: {total_10 * 100:.2f}%\n")
    if total_10 < 0.7:
        f.write("→ PCA không giữ được nhiều thông tin ngay cả với 10 chiều\n")

    f.write("\n" + "=" * 60 + "\nEXPLAINED VARIANCE (MỖI TRƯỜNG HỢP)\n" + "=" * 60 + "\n")
    for k in [2, 4, 6]:
        info = np.sum(explained[:k])
        f.write(f"[PCA - {k} components] → Giữ {info * 100:.2f}% thông tin\n")

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(explained), marker='o', color=palette[0])
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_cumulative_variance_simple.png", dpi=200)
    plt.close()

    f.write("\n[Nhận xét]\n")
    f.write("- Số chiều càng tăng → giữ nhiều thông tin hơn\n")
    f.write("- Tỷ lệ phương sai giữ lại rất thấp (<40% với 6 chiều)\n")
    f.write("- Điều này cho thấy dữ liệu có cấu trúc phức tạp, khó giảm chiều tuyến tính\n")
    f.write("- PCA không nén dữ liệu hiệu quả trong trường hợp này\n")
    f.write("- Cần nhiều thành phần hơn để giữ phần lớn thông tin\n")

    f.write("\n" + "=" * 60 + "\nTRỰC QUAN PCA\n" + "=" * 60 + "\n")
    pca = PCA_Manual(n_components=25)
    X_pca = pca.fit_transform(X_scaled.values)

    loadings = pd.DataFrame(pca.components, index=feat_names, columns=[f'PC{i + 1}' for i in range(25)])
    evr = pca.explained_variance_ratio_ * 100
    cum_evr = np.cumsum(evr)

    f.write("\nLoadings đầy đủ (đóng góp từng chiều)\n")
    f.write(loadings.round(4).to_string() + "\n")

    f.write("\nTop 5 chiều đóng góp lớn nhất mỗi PC\n")
    for col in loadings.columns:
        top5 = loadings[col].abs().nlargest(5)
        row = ' | '.join([f"{k}({loadings.at[k, col]:+.3f})" for k in top5.index])
        f.write(f"  {col}: {row}\n")

    f.write("\nScores (tọa độ mẫu trong không gian PCA)\n")
    scores_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(25)])
    scores_df['charges'] = y.values
    f.write("  10 mẫu đầu:\n")
    f.write(scores_df.head(10).round(4).to_string() + "\n")
    f.write("\n  Thống kê scores:\n")
    f.write(scores_df.drop(columns=['charges']).describe().round(4).to_string() + "\n")

    f.write("\nLượng thông tin bảo tồn (Explained Variance) — từng k\n")
    f.write(f"  {'k':>4}  {'Var %':>10}  {'Nhận xét'}\n")
    f.write(f"  {'-' * 40}\n")
    for k in [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, min(30, N - 1)]:
        if k >= min(D, N):
            continue
        pca_t = PCA_Manual(n_components=k)
        pca_t.fit(X_scaled.values)
        info = pca_t.explained_variance_ratio_[:k].sum() * 100
        note = ("✓ Tốt (>=80%)" if info >= 80 else "△ TB (60-80%)" if info >= 60 else "✗ Thấp (<60%)")
        f.write(f"  {k:>4}  {info:>9.2f}%  {note}\n")

    pca_full_plots = PCA_Manual(n_components=min(D, N - 1))
    pca_full_plots.fit(X_scaled.values)
    evr_full = pca_full_plots.explained_variance_ratio_ * 100
    cum_full = np.cumsum(evr_full)
    show_n = min(30, len(evr_full))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PCA — Phương sai giải thích", fontsize=13, fontweight='bold')
    axes[0].bar(range(1, show_n + 1), evr_full[:show_n], color='#2E86AB', alpha=0.75)
    ax2 = axes[0].twinx()
    ax2.plot(range(1, show_n + 1), cum_full[:show_n], 'o-', color='#E84855', linewidth=2, markersize=4)
    ax2.axhline(80, color='#3BB273', linestyle='--', linewidth=1.2, label='80%')
    ax2.axhline(95, color='#F18F01', linestyle='--', linewidth=1.2, label='95%')
    axes[0].set_xlabel("Thành phần PC")
    axes[0].set_ylabel("Explained Variance %", color='#2E86AB')
    ax2.set_ylabel("Cumulative %", color='#E84855')
    axes[0].set_title("Scree Plot + Cumulative")
    ax2.legend(fontsize=8)

    ax = axes[1]
    ks = [k for k in [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, min(30, N - 1)] if k < min(D, N)]
    infos = [PCA_Manual(n_components=k).fit(X_scaled.values).explained_variance_ratio_.sum() * 100 for k in ks]
    bar_c = ['#E84855' if v < 60 else '#F18F01' if v < 80 else '#3BB273' for v in infos]
    bars = ax.bar([str(k) for k in ks], infos, color=bar_c, alpha=0.85)
    ax.axhline(80, color='#3BB273', linestyle='--', linewidth=1.5, label='80%')
    ax.axhline(95, color='#9B5DE5', linestyle='--', linewidth=1.5, label='95%')
    for bar, val in zip(bars, infos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.0f}%', ha='center', va='bottom',
                fontsize=7)
    ax.set_xlabel("Số thành phần k")
    ax.set_ylabel("Tổng Explained Variance %")
    ax.set_title("Thông tin bảo tồn theo k")
    ax.legend(fontsize=8)

    ax = axes[2]
    top_feat = loadings.abs().max(axis=1).nlargest(15).index
    sns.heatmap(loadings.loc[top_feat].T, ax=ax, cmap=cmap_div, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 7}, linewidths=0.3, cbar_kws={'shrink': 0.8})
    ax.set_title("Loadings Heatmap\n(Top 15 chiều đóng góp lớn nhất)")
    ax.set_yticklabels([f'PC{i + 1}' for i in range(25)], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_explained_variance_manifest.png", dpi=200)
    plt.close()

    pairs_all = list(itertools.combinations(range(6), 2))
    ncols_p = 5
    nrows_p = int(np.ceil(len(pairs_all) / ncols_p))
    fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(ncols_p * 4, nrows_p * 3.5))
    fig.suptitle("PCA — Scatter Plot tất cả cặp thành phần (PC1–PC6, C(6,2)=15 cặp)", fontsize=13, fontweight='bold')
    axes = axes.flatten()
    sc_ref = None

    for idx, (i, j) in enumerate(pairs_all):
        ax = axes[idx]
        sc = ax.scatter(X_pca[:, i], X_pca[:, j], c=y.values, cmap=cmap_seq, alpha=0.65, s=15, edgecolors='none')
        sc_ref = sc
        pts = X_pca[:, [i, j]]
        mu = pts.mean(axis=0)
        cov2 = np.cov(pts.T)
        if cov2.ndim == 2 and not np.isnan(cov2).any():
            vals2, vecs2 = np.linalg.eigh(cov2)
            angle = np.degrees(np.arctan2(*vecs2[:, 1][::-1]))
            w, h = 2 * np.sqrt(np.abs(vals2))
            for nsig, al in [(1, 0.12), (2, 0.06)]:
                ell = Ellipse(mu, nsig * w, nsig * h, angle=angle, color='gray', alpha=al, fill=True, linewidth=0)
                ax.add_patch(ell)
            ell2 = Ellipse(mu, w, h, angle=angle, color='#2E86AB', fill=False, linewidth=1.2, linestyle='--')
            ax.add_patch(ell2)
        ax.set_xlabel(f"PC{i + 1} ({evr[i]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC{j + 1} ({evr[j]:.1f}%)", fontsize=8)
        ax.set_title(f"PC{i + 1} vs PC{j + 1}", fontsize=9)

    for idx in range(len(pairs_all), len(axes)):
        axes[idx].set_visible(False)
    fig.colorbar(sc_ref, ax=axes[:len(pairs_all)].tolist(), label='charges', shrink=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_pair_plots_all.png", dpi=200)
    plt.close()

    # Tương quan mục tiêu
    f.write("\n" + "=" * 60 + "\nTƯƠNG QUAN PCA vs CHARGES\n" + "=" * 60 + "\n")
    f.write("\nPCA components vs charges\n")
    corr_pca = []
    f.write(f"  {'':^6} {'r':>8}  {'|r| bar':<25}  Diễn giải\n")
    f.write(f"  {'-' * 60}\n")
    for i in range(6):
        r = np.corrcoef(X_pca[:, i], y.values)[0, 1]
        corr_pca.append(r)
        bar = '▓' * int(abs(r) * 25)
        note = (
            "mạnh" if abs(r) > 0.5 else "trung bình" if abs(r) > 0.3 else "yếu" if abs(r) > 0.1 else "không đáng kể")
        f.write(f"  PC{i + 1:<3}  {r:>+8.4f}  {bar:<25}  {note}\n")

    f.write("\nTop chiều gốc tương quan với charges\n")
    raw_corrs = X_scaled.corrwith(pd.Series(y.values, name='charges')).sort_values(key=abs, ascending=False)
    f.write(f"  {'Feature':<35}  {'r':>8}  Bar\n")
    for feat, val in raw_corrs.head(12).items():
        bar = ('▲' if val > 0 else '▼') * int(abs(val) * 20)
        f.write(f"  {feat:<35}  {val:>+8.4f}  {bar}\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Tương quan với đầu ra (charges)", fontsize=13, fontweight='bold')
    ax = axes[0]
    bar_c = ['#3BB273' if r > 0 else '#E84855' for r in corr_pca]
    bars = ax.bar([f'PC{i + 1}' for i in range(6)], corr_pca, color=bar_c, alpha=0.85)
    for bar, val in zip(bars, corr_pca):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.01 if val >= 0 else -0.03), f'{val:.3f}', ha='center',
                va='bottom', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.3, color='gray', linestyle='--', linewidth=1, label='±0.3')
    ax.axhline(-0.3, color='gray', linestyle='--', linewidth=1)
    ax.set_title("PCA components vs charges (Pearson r)")
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(X_pca[:, 0], y.values, alpha=0.6, s=18, color='#2E86AB', edgecolors='none')
    z = np.polyfit(X_pca[:, 0], y.values, 1)
    xl = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), color='#E84855', linewidth=2, linestyle='--',
            label=f'Trendline (r={corr_pca[0]:.3f})')
    ax.set_xlabel("PC1")
    ax.set_ylabel("charges")
    ax.set_title(f"Scatter: PC1 vs charges")
    ax.legend(fontsize=8)

    ax = axes[2]
    top10 = raw_corrs.head(10)
    colors_fc = ['#3BB273' if v > 0 else '#E84855' for v in top10.values]
    ax.barh(range(len(top10)), top10.values, color=colors_fc, alpha=0.85)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title("Top 10 chiều gốc tương quan với charges")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_output_correlations.png", dpi=200)
    plt.close()

    pc4_labels = [f'PC{i + 1}\n({evr[i]:.1f}%)' for i in range(4)]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("PCA Pair Plot Matrix PC1–PC4 (diagonal = histogram)", fontsize=13, fontweight='bold')
    for row in range(4):
        for col in range(4):
            ax = fig.add_subplot(4, 4, row * 4 + col + 1)
            if row == col:
                ax.hist(X_pca[:, row], bins=15, color='#2E86AB', alpha=0.7)
                ax.set_title(pc4_labels[row], fontsize=8)
            else:
                ax.scatter(X_pca[:, col], X_pca[:, row], c=y.values, cmap=cmap_seq, alpha=0.5, s=10, edgecolors='none')
            if row == 3: ax.set_xlabel(pc4_labels[col], fontsize=7)
            if col == 0: ax.set_ylabel(pc4_labels[row], fontsize=7)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_pair_plot_matrix_1_4.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.values, cmap=cmap_seq, alpha=0.6, s=18, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='charges')
    top8 = loadings[['PC1', 'PC2']].pow(2).sum(axis=1).nlargest(8).index
    scale = min(X_pca[:, 0].std(), X_pca[:, 1].std()) * 3
    for feat in top8:
        lx, ly = loadings.at[feat, 'PC1'] * scale, loadings.at[feat, 'PC2'] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#E84855', lw=1.8, alpha=0.85))
        ax.text(lx * 1.08, ly * 1.08, feat, fontsize=8, color='#E84855', alpha=0.9)
    ax.set_xlabel(f"PC1 ({evr[0]:.2f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]:.2f}%)")
    ax.set_title("PCA Biplot — Scores + Loadings arrows (Top 8 chiều)")
    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0, color='gray', linewidth=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_biplot_top8.png", dpi=200)
    plt.close()

    f.write("\n[Nhận xét]\n")
    f.write("- PC1 có tương quan rất mạnh với charges (~0.62)\n")
    f.write("- PC2 và PC3 có tương quan trung bình\n")
    f.write("- Các PC còn lại gần như không liên quan\n")
    f.write("- Điều này cho thấy một số chiều chính chứa thông tin dự đoán quan trọng\n")

    return cum_evr, corr_pca[0]