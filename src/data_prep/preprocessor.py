import logging
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Pipeline xử lý dữ liệu từ bước làm sạch, mã hóa danh mục đến chuẩn hóa số số học."""

    def __init__(self):
        self.scale_params = {}
        self.leakage_cols = ['death', 'hospdead', 'sfdm2', 'totcst', 'totmcst', 'surv2m', 'surv6m', 'prg2m', 'prg6m']
        self.valid_ranges = {
            'age': (18, 120), 'meanbp': (10, 300), 'temp': (25, 42),
            'hrt': (20, 300), 'resp': (4, 60), 'ph': (6.5, 8.0), 'sod': (100, 180)
        }
        self.harrell_fill = {
            'alb': 3.5, 'pafi': 333.3, 'bili': 1.01, 'crea': 1.01,
            'bun': 6.51, 'wblc': 9.0, 'urine': 2502.0
        }
        self.binary_map = {
            'sex': {'male': 0, 'female': 1},
            'ca': {'no': 0, 'yes': 1, 'metastatic': 2}
        }
        self.ordinal_map = {
            'income': {'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}
        }
        self.ohe_cols = ['race', 'dzgroup', 'dzclass', 'dnr']

    def clean_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu thô: lọc rò rỉ, xử lý giá trị ngoài khoảng y khoa và điền khuyết."""
        logger.info("Khởi chạy tiến trình làm sạch dữ liệu (Data Cleaning)...")
        df = df_raw.copy()

        # Lọc bỏ các hàng khuyết biến mục tiêu
        df = df.dropna(subset=['charges'])

        # Loại bỏ các cột gây rò rỉ thông tin (Data Leakage)
        cols_to_drop = [c for c in self.leakage_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Loại bỏ các bản ghi trùng lặp
        df = df.drop_duplicates(keep='first').reset_index(drop=True)

        # Chuẩn hóa định dạng chuỗi ký tự văn bản
        if 'sex' in df.columns:
            df['sex'] = df['sex'].astype(str).str.strip()
        if 'ca' in df.columns:
            df['ca'] = df['ca'].astype(str).str.strip()

        for col in ['race', 'income', 'dzgroup', 'dzclass']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().replace(
                    {'nan': np.nan, 'none': np.nan, '': np.nan})

        # Xử lý các giá trị bất thường ngoài giới hạn sinh lý con người
        for col, (lo, hi) in self.valid_ranges.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
                if mask.sum():
                    df.loc[mask, col] = np.nan

        # Điền khuyết theo khuyến nghị chuyên gia (Prof. Frank Harrell)
        for col, val in self.harrell_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # Điền khuyết cho các cột còn lại bằng các chỉ số thống kê (Mean/Median/Mode)
        for col in df.columns:
            if col == 'charges' or df[col].isna().sum() == 0:
                continue
            if df[col].dtype in ['float64', 'int64']:
                fill_val = df[col].median() if abs(df[col].skew()) > 1.0 else df[col].mean()
            else:
                mode_s = df[col].mode()
                fill_val = mode_s[0] if len(mode_s) else 'unknown'
            df[col] = df[col].fillna(fill_val)

        # Đảm bảo đúng định dạng kiểu dữ liệu hệ thống
        for col in ['age', 'num.co']:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        for col in ['sex', 'race', 'income', 'dzgroup', 'dzclass', 'ca']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        logger.info(f"Hoàn tất làm sạch dữ liệu. Kích thước hiện tại: {df.shape}")
        return df

    def encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuyển đổi các biến phân loại định tính sang dạng số học để mô hình hóa."""
        logger.info("Bắt đầu tiến trình mã hóa biến định tính (Categorical Encoding)...")
        df_enc = df.copy()

        for col, mapping in self.binary_map.items():
            if col in df_enc.columns:
                df_enc[col] = pd.to_numeric(df_enc[col].map(mapping), errors='coerce')

        for col, mapping in self.ordinal_map.items():
            if col in df_enc.columns:
                df_enc[col] = pd.to_numeric(df_enc[col].map(mapping), errors='coerce')

        existing_ohe = [c for c in self.ohe_cols if c in df_enc.columns]
        for col in existing_ohe:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True, dtype=int)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)

        logger.info(f"Mã hóa danh mục hoàn tất. Kích thước hiện tại: {df_enc.shape}")
        return df_enc

    def scale_data(self, df_enc: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa phân phối các đặc trưng liên tục nhằm tối ưu thuật toán hồi quy."""
        logger.info("Áp dụng biến đổi Log và chuẩn hóa phân phối Z-score...")
        df_scaled = df_enc.copy()

        # Giảm hiện tượng lệch phải nặng bằng phép biến đổi log1p
        log_cols = ['charges', 'bili', 'crea', 'bun', 'wblc', 'urine', 'sps', 'aps']
        for col in [c for c in log_cols if c in df_scaled.columns]:
            df_scaled[col] = np.log1p(df_scaled[col].clip(lower=0))

        # Phân lập các biến không tham gia chuẩn hóa Z-score
        binary_flags = [c for c in df_scaled.columns if
                        df_scaled[c].nunique() == 2 and df_scaled[c].min() == 0 and df_scaled[c].max() == 1]
        ohe_new_cols = [c for c in df_scaled.columns if any(c.startswith(p + '_') for p in self.ohe_cols)]
        skip_scale = set(['charges'] + binary_flags + ohe_new_cols + ['income', 'ca'])
        scale_cols = [c for c in df_scaled.select_dtypes(include='number').columns if c not in skip_scale]

        # Thực thi Standard Z-score
        for col in scale_cols:
            mu, sig = df_scaled[col].mean(), df_scaled[col].std()
            if sig == 0:
                continue
            df_scaled[col] = (df_scaled[col] - mu) / sig
            self.scale_params[col] = {'mean': round(float(mu), 4), 'std': round(float(sig), 4)}

        # Xuất file cấu hình phục vụ giai đoạn Inference sau này
        with open('scale_params.json', 'w') as f:
            json.dump(self.scale_params, f, indent=2)

        logger.info("Lưu trữ bộ tham số scale_params.json thành công.")
        return df_scaled