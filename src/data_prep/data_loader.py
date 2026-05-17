import logging
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Cấu hình logging cơ bản cho module
logger = logging.getLogger(__name__)


class DataLoader:
    """Class đảm nhận nhiệm vụ kết nối và tải dữ liệu từ các kho lưu trữ trực tuyến."""

    def __init__(self, repo_id: int = 880):
        self.repo_id = repo_id

    def fetch_data(self) -> pd.DataFrame:
        """Tải dữ liệu gốc từ UCI Machine Learning Repository dựa trên ID.

        Returns:
            pd.DataFrame: Toàn bộ tập dữ liệu thô bao gồm cả features và targets.
        """
        logger.info(  # Chạy luồng tải dữ liệu
            f"Bắt đầu tải dữ liệu từ UCI Repo (ID: {self.repo_id})..."
        )
        try:
            support2 = fetch_ucirepo(id=self.repo_id)
            df_raw = pd.concat([support2.data.features, support2.data.targets], axis=1)
            logger.info(f"Tải dữ liệu thành công. Kích thước gốc: {df_raw.shape}")
            return df_raw
        except Exception as e:
            logger.error(f"Lỗi xảy ra khi tải dữ liệu từ UCI: {str(e)}")
            raise e