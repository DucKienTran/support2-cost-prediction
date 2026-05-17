import pandas as pd
import numpy as np


class DataSummaryReporter:
    """Module xuất báo cáo thống kê dạng text cho dữ liệu sau chuẩn hóa."""

    @staticmethod
    def print_post_scale_summary(df_scaled: pd.DataFrame, target_col: str):
        SEP = "─" * 68
        print(f"\n{SEP}")
        print("  MÔ TẢ DỮ LIỆU SAU CHUẨN HÓA")
        print(SEP)

        # 1. Thông tin tổng quan (Shape, Missing, Dtypes)
        print(f"\n▸ Shape  : {df_scaled.shape[0]:,} hàng × {df_scaled.shape[1]} cột")
        print(f"▸ Missing: {df_scaled.isna().sum().sum()} (phải = 0)")

        # Nhóm đếm dtypes thành dict gọn gàng
        dtype_counts = df_scaled.dtypes.astype(str).value_counts().to_dict()
        print(f"▸ Dtypes : {dtype_counts}")

        # Tự động phân loại cột OHE (chỉ chứa 0 và 1) và cột Numeric đã scale
        ohe_cols = [c for c in df_scaled.columns if set(df_scaled[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        scale_cols = [c for c in df_scaled.select_dtypes(include=np.number).columns if
                      c not in ohe_cols and c != target_col]

        # 2. Thống kê mô tả sau scale (chọn tối đa 12 cột đại diện)
        sample_scaled = scale_cols[:12]
        if sample_scaled:
            print(f"\n Thống kê mô tả sau chuẩn hóa (chọn {len(sample_scaled)} cột đại diện):")
            desc_scaled = df_scaled[sample_scaled].describe().T
            print(desc_scaled[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3).to_string())

        print("\n  → Sau Z-score: mean ≈ 0, std ≈ 1 với tất cả cột số (đặc trưng X)")
        print("  → OHE columns vẫn là 0/1 nguyên")
        print("  → ORDINAL columns giữ nguyên")
        print(f"  → Target ('{target_col}') ở dạng Log1p")

        # 3. Thống kê cột One-hot Encoding
        if ohe_cols:
            print(f"\n Cột One-Hot Encoding (tỷ lệ = 1):")
            for col in ohe_cols:
                pct = df_scaled[col].mean() * 100
                print(f"  {col:<40}: {pct:5.1f}%")

        # 4. Phân phối biến target
        if target_col in df_scaled.columns:
            print(f"\n Phân phối biến mục tiêu ({target_col} - dạng Log1p):")
            bins = pd.cut(df_scaled[target_col], bins=5)
            print(bins.value_counts().sort_index().to_string())

        print(f"{SEP}\n")