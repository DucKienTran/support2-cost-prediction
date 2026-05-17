import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd


class EdaDataVisualizer:
    """Class trực quan hóa dữ liệu sử dụng bộ Dashboard tùy chỉnh gốc của dự án."""

    @staticmethod
    def generate_plots(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame, df_scaled: pd.DataFrame,
                       output_dir: str = "reports/plots"):
        """Kết xuất bảng điều khiển tổng hợp 8 biểu đồ Support2 EDA."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Định nghĩa lại bộ màu C gốc (do snippet bị thiếu)
        C = {
            'bg': '#F8F9FA', 'dark': '#212529', 'red': '#E63946',
            'amber': '#F4A261', 'blue': '#1D3557', 'green': '#2A9D8F',
            'teal': '#457B9D', 'purple': '#7209B7', 'gray': '#6C757D'
        }

        # 2. Khớp tên biến từ pipeline vào code gốc của bạn
        df = df_cleaned

        # 3. Tự động tìm các cột One-Hot Encoding (có giá trị chỉ 0 và 1) trong df_scaled
        ohe_existing = [col for col in df_scaled.columns if
                        set(df_scaled[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})]

        # =========================================================================
        # TOÀN BỘ CODE VISUALIZE BÊN DƯỚI LÀ NGUYÊN VĂN TỪ BẢN GỐC CỦA BẠN
        # =========================================================================
        fig = plt.figure(figsize=(22, 26), facecolor=C['bg'])
        fig.suptitle("SUPPORT2 — Phân tích & Chuẩn hóa Dữ liệu (Mục tiêu: charges)", fontsize=17,
                     fontweight='bold', color=C['dark'], y=0.99)

        gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.38)

        # (1) Tỷ lệ missing gốc
        ax1 = fig.add_subplot(gs[0, :2])
        miss = (df_raw.isna().sum() / len(df_raw) * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        colors_bar = [C['red'] if v > 30 else C['amber'] if v > 10 else C['blue'] for v in miss]
        bars = ax1.barh(miss.index[::-1], miss.values[::-1], color=colors_bar[::-1], alpha=0.85)
        ax1.axvline(50, color=C['red'], ls='--', lw=1.2, alpha=0.7, label='Ngưỡng xóa cột (50%)')
        ax1.set_xlabel("Tỷ lệ thiếu (%)")
        ax1.set_title("Tỷ lệ giá trị thiếu (dữ liệu gốc)", fontweight='bold')
        ax1.legend(fontsize=8)
        for bar, val in zip(bars, miss.values[::-1]):
            ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=7.5)

        # (2) Phân phối Ung thư (ca) thay vì death
        ax2 = fig.add_subplot(gs[0, 2])
        if 'ca' in df.columns:
            ca_vc = df['ca'].value_counts()
            ax2.pie(ca_vc, labels=ca_vc.index, colors=[C['green'], C['amber'], C['red']],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 10})
            ax2.set_title("Phân phối ung thư (ca)", fontweight='bold')

        # (3) Trước/Sau log transform cho charges
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        if 'charges' in df_raw.columns:
            raw_ch = df_raw['charges'].dropna()
            log_ch = np.log1p(raw_ch.clip(lower=0))
            ax3.hist(raw_ch, bins=60, color=C['amber'], edgecolor='white', alpha=0.85)
            ax3.set_title(f"charges (gốc, skew={raw_ch.skew():.2f})", fontweight='bold')
            ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x / 1000:.0f}K'))
            ax4.hist(log_ch, bins=60, color=C['teal'], edgecolor='white', alpha=0.85)
            ax4.set_title(f"charges sau log1p (skew={log_ch.skew():.2f})", fontweight='bold')

        # (4) Trước/Sau Z-score (age)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'age' in df.columns:
            age_raw = df['age'].dropna()
            age_z = (age_raw - age_raw.mean()) / age_raw.std()
            ax5.hist(age_raw, bins=35, color=C['purple'], edgecolor='white', alpha=0.6, label='Gốc')
            ax5_r = ax5.twinx()
            ax5_r.hist(age_z, bins=35, color=C['blue'], edgecolor='white', alpha=0.5, label='Z-score')
            ax5.set_title("age: gốc vs Z-score", fontweight='bold')
            ax5.set_xlabel("Tuổi / Z-score")
            lines1, _ = ax5.get_legend_handles_labels()
            lines2, _ = ax5_r.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, ['Gốc', 'Z-score'], fontsize=8, loc='upper left')

        # (5) Correlation heatmap (Đã dọn dẹp các biến leakage)
        ax6 = fig.add_subplot(gs[2, :2])
        corr_cols = ['age', 'meanbp', 'hrt', 'resp', 'temp', 'charges', 'los', 'bun', 'crea', 'wblc']
        corr_cols = [c for c in corr_cols if c in df.columns]
        if corr_cols:
            corr = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, ax=ax6, mask=mask, annot=True, fmt='.2f',
                        cmap='RdBu_r', center=0, linewidths=0.5,
                        annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
            ax6.set_title("Ma trận tương quan (Loại bỏ rò rỉ dữ liệu)", fontweight='bold')
            ax6.tick_params(axis='x', rotation=40, labelsize=8)
            ax6.tick_params(axis='y', rotation=0, labelsize=8)

        # (6) Phân phối nhóm bệnh
        ax7 = fig.add_subplot(gs[2, 2])
        if 'dzgroup' in df.columns:
            dz = df['dzgroup'].value_counts()
            ax7.barh(dz.index[::-1], dz.values[::-1], color=C['purple'], alpha=0.8)
            ax7.set_title("Phân phối nhóm bệnh (dzgroup)", fontweight='bold')
            ax7.set_xlabel("Số bệnh nhân")

        # (7) Boxplot Z-score sau scale
        ax8 = fig.add_subplot(gs[3, :])
        box_cols = [c for c in ['age', 'meanbp', 'hrt', 'resp', 'alb', 'sod', 'ph'] if c in df_scaled.columns]
        if box_cols:
            box_data = [df_scaled[c].dropna().values for c in box_cols]
            bp = ax8.boxplot(box_data, labels=box_cols, patch_artist=True,
                             medianprops={'color': C['red'], 'linewidth': 2},
                             flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
            # Lặp palette an toàn để không bị lỗi nếu số lượng cột thay đổi
            palette = [C['blue'], C['teal'], C['purple'], C['amber'], C['green'], C['red'], C['gray']] * 3
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color);
                patch.set_alpha(0.65)
            ax8.axhline(0, color=C['dark'], ls='--', lw=0.8, alpha=0.5, label='mean=0')
            ax8.axhline(1, color=C['gray'], ls=':', lw=0.8, alpha=0.5, label='±1 std')
            ax8.axhline(-1, color=C['gray'], ls=':', lw=0.8, alpha=0.5)
            ax8.set_title("Phân phối sau Z-score Standardization (mean≈0, std≈1)", fontweight='bold')
            ax8.set_ylabel("Z-score");
            ax8.legend(fontsize=8)

        # (8) OHE proportion bar chart
        ax9 = fig.add_subplot(gs[4, :])
        ohe_pct = {c: df_scaled[c].mean() * 100 for c in ohe_existing if c in df_scaled.columns}
        if ohe_pct:
            ohe_series = pd.Series(ohe_pct).sort_values(ascending=False)
            ax9.bar(range(len(ohe_series)), ohe_series.values, color=C['teal'], alpha=0.8, edgecolor='white')
            ax9.set_xticks(range(len(ohe_series)))
            ax9.set_xticklabels(ohe_series.index, rotation=40, ha='right', fontsize=8)
            ax9.set_ylabel("Tỷ lệ = 1 (%)")
            ax9.set_title("Tỷ lệ giá trị 1 trong các cột One-Hot Encoding", fontweight='bold')

        # Cập nhật lưu file về thư mục output_dir thay vì Google Drive
        save_path = os.path.join(output_dir, 'support2_full_analysis.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=C['bg'])
        plt.close()