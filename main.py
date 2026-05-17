import matplotlib

matplotlib.use('Agg')

import os
import logging
import warnings
import pandas as pd

# Tắt tất cả các thông báo Warning rác từ hệ thống
warnings.filterwarnings("ignore")

from src.data_prep.preprocessor import DataPreprocessor
from src.data_prep.eda_visualizer import EdaDataVisualizer
from src.data_prep.data_summary import DataSummaryReporter
from src.dim_reduction.dim_reduction import DimensionalityReductionPipeline


def main():
    # Định dạng hiển thị log sạch sẽ, không chặn dọc
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    SEP = "─" * 70

    print(SEP)
    logger.info("KHỞI CHẠY PIPELINE PHÂN TÍCH DỮ LIỆU SUPPORT2")
    print(SEP)

    data_path = "data/raw/support2_raw.csv"
    base_report_dir = "reports"  # Thư mục gốc chứa toàn bộ báo cáo

    if not os.path.exists(data_path):
        logger.error(f"Thất bại - Không tìm thấy tệp dữ liệu tại: {data_path}")
        print(SEP)
        return

    try:
        # 0. ĐỌC DỮ LIỆU
        logger.info(f"Đọc dữ liệu - Nguồn: {data_path}")
        df_raw = pd.read_csv(data_path)

        processor = DataPreprocessor()

        # 1. TIỀN XỬ LÝ (CLEANING)
        logger.info("Tiền xử lý  - Đang làm sạch và khử nhiễu Outliers...")
        df_cleaned = processor.clean_data(df_raw)

        # 2. MÃ HÓA (ENCODING)
        logger.info("Mã hóa data - Chuyển đổi các đặc trưng Categorical...")
        df_encoded = processor.encode_data(df_cleaned)

        # 3. CHUẨN HÓA (SCALING)
        logger.info("Chuẩn hóa   - Áp dụng biến đổi Z-score toàn không gian...")
        df_scaled = processor.scale_data(df_encoded)

        # 4. ĐỒ THỊ EDA (Đẩy vào thư mục 01_eda)
        eda_dir = os.path.join(base_report_dir, "01_eda")
        logger.info("Đồ thị EDA  - Đang kết xuất Dashboard vào thư mục '01_eda'...")
        EdaDataVisualizer.generate_plots(df_raw, df_cleaned, df_scaled, output_dir=eda_dir)

        # 5. IN BÁO CÁO THỐNG KÊ (In ra Terminal)
        DataSummaryReporter.print_post_scale_summary(df_scaled, target_col="charges")

        # 6. GIẢM CHIỀU (Đẩy vào thư mục 02_dim_reduction)
        dim_dir = os.path.join(base_report_dir, "02_dim_reduction")
        logger.info("Giảm chiều  - Khởi động PCA & UMAP, kết quả lưu vào '02_dim_reduction'...")
        dr_pipeline = DimensionalityReductionPipeline(df_scaled=df_scaled, task_type="regression")
        dr_pipeline.run(output_dir=dim_dir)

        print(SEP)
        logger.info("HOÀN TẤT    - Toàn bộ pipeline đã thực thi thành công!")
        logger.info(f"Kết quả     - Xem chi tiết tại thư mục cấu trúc: '{base_report_dir}/'")
        print(SEP)

    except Exception as e:
        print(SEP)
        logger.critical(f"SỰ CỐ KHẨN  - Luồng vận hành bị dừng gãy. Lỗi: {str(e)}")
        print(SEP)


if __name__ == "__main__":
    main()