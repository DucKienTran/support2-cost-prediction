import os
import logging
import matplotlib.pyplot as plt
from .pca_analysis import run_pca_pipeline
from .umap_analysis import run_umap_pipeline

logger = logging.getLogger(__name__)


class DimensionalityReductionPipeline:
    """Hệ thống điều phối cốt lõi gọi các module giảm chiều độc lập và in bảng so sánh kỹ thuật."""

    def __init__(self, df_scaled, task_type="regression"):
        self.df = df_scaled.copy()
        self.task_type = task_type
        self.X_scaled = df_scaled.drop(columns=["charges"])
        self.y = df_scaled["charges"]
        self.feat_names = list(self.X_scaled.columns)
        self.N, self.D = self.X_scaled.shape

        # Thiết lập style đồ họa toàn cục
        plt.rcParams.update({
            'figure.facecolor': '#F8F9FA',
            'axes.facecolor': '#FFFFFF',
            'axes.grid': True,
            'grid.alpha': 0.25,
            'grid.linestyle': '--',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        self.palette = ['#2E86AB', '#E84855', '#3BB273', '#F18F01', '#9B5DE5', '#00B4D8']
        self.cmap_div = 'RdBu_r'
        self.cmap_seq = 'viridis'

    def run(self, output_dir="reports"):
        """Khởi chạy toàn bộ hệ thống giảm chiều và phân phối kết quả vào các thư mục con."""

        # 1. Tự động định nghĩa và tạo các thư mục con phân cấp
        pca_dir = os.path.join(output_dir, "pca")
        umap_dir = os.path.join(output_dir, "umap")
        lda_dir = os.path.join(output_dir, "lda")  # Sẵn sàng chỗ trống cho LDA sau này

        os.makedirs(pca_dir, exist_ok=True)
        os.makedirs(umap_dir, exist_ok=True)
        os.makedirs(lda_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "detailed_outputs.txt")
        logger.info(f"Bắt đầu xử lý giảm chiều phối hợp cho bài toán: {self.task_type.upper()}")

        with open(report_path, "w", encoding="utf-8") as f:
            # SỬA TẠI ĐÂY: Truyền thẳng `pca_dir` vào thay vì `output_dir`
            cum_evr, pc1_corr = run_pca_pipeline(
                self.X_scaled, self.y, self.feat_names, self.N, self.D,
                self.cmap_div, self.cmap_seq, self.palette, f, pca_dir
            )

            # SỬA TẠI ĐÂY: Truyền thẳng `umap_dir` vào thay vì `output_dir`
            run_umap_pipeline(self.X_scaled, self.y, self.N, self.cmap_seq, f, umap_dir)

            # 3. Kết xuất bảng so sánh phương pháp vào file txt tổng
            f.write("\n" + "=" * 60 + "\nSO SÁNH PHƯƠNG PHÁP\n" + "=" * 60 + "\n")
            rows = [
                ("Loại", "Tuyến tính", "Phi tuyến"),
                ("Cấu trúc giữ lại", "Toàn cục (phương sai)", "Cục bộ (topology)"),
                ("Explained Variance", f"{cum_evr[5] * 100:.1f}% (6 PC)", "Không đo trực tiếp"),
                ("Diễn giải components", "Có (loadings rõ ràng)", "Khó diễn giải"),
                ("Tái tạo dữ liệu", "Có thể chính xác", "Không đảm bảo"),
                ("Tốc độ", "Nhanh O(D²N)", "Chậm O(N²)"),
                ("Tính ổn định", "Cao (deterministic)", "Phụ thuộc params"),
                ("Phù hợp với", "Dữ liệu tuyến tính", "Cấu trúc phức tạp"),
                ("Corr PC1 vs charges", f"r = {pc1_corr:+.4f}", "Không áp dụng"),
                ("Ứng dụng", "Feature eng., reconstruction", "Visualization"),
            ]
            f.write(f"  {'Tiêu chí':<28}  {'PCA':^28}  {'UMAP':^28}\n")
            f.write(f"  {'-' * 88}\n")
            for r0, r1, r2 in rows:
                f.write(f"  {r0:<28}  {r1:<28}  {r2:<28}\n")

        # Cập nhật log hiển thị đường dẫn cho đúng thực tế
        logger.info(f"Hoàn tất! Chi tiết so sánh tại: '{report_path}'")
        logger.info(f"Đã phân tách cụm ảnh thuật toán vào các thư mục tương ứng trong: '{output_dir}/'")