import pandas as pd
import numpy as np
import re
import os


class SkincareDataCleaner:
    """Data cleaning with step-by-step output"""

    def __init__(self):
        self.output_dir = 'dataset/processed/'
        os.makedirs(self.output_dir, exist_ok=True)

    # CLEAN BRAND
    def clean_brand_name(self, brand_str):

        if pd.isna(brand_str):
            return "Unknown"

        brand_str = str(brand_str)

        # remove leading number
        brand_str = re.sub(r'^\d+_', '', brand_str)

        parts = brand_str.split('_')

        clean_parts = []

        for part in parts:

            if len(part) > 2 and not part.isdigit():

                clean_part = re.sub(r'[^a-zA-Z\s]', '', part)

                if len(clean_part) > 2:
                    clean_parts.append(clean_part)

        if clean_parts:
            return clean_parts[0].capitalize()[:20]

        return "Unknown"

    # CLEAN CATEGORY
    def clean_category_name(self, category_str):

        if pd.isna(category_str):
            return "Other"

        category_str = str(category_str)

        category_str = re.sub(r'[{}]', '', category_str)
        category_str = re.sub(r'[^\w\s]', '', category_str)
        category_str = category_str.strip()

        if 'face wash' in category_str.lower():
            return 'Face Wash'

        elif 'face cream' in category_str.lower():
            return 'Face Cream'

        elif 'serum' in category_str.lower():
            return 'Face Serum'

        elif 'body' in category_str.lower():
            return 'Body Care'

        elif len(category_str) > 0:
            return category_str.title()[:30]

        return "Other"


    # CLEAN PRICE
    def extract_price(self, price_str):

        if pd.isna(price_str):
            return 50000

        try:
            # convert to string
            cleaned = str(price_str)

            # remove Rp, spaces, dots
            cleaned = cleaned.replace('Rp', '')
            cleaned = cleaned.replace('rp', '')
            cleaned = cleaned.replace('.', '')
            cleaned = cleaned.replace(' ', '')

            # extract all numbers
            numbers = re.findall(r'\d+', cleaned)

            if len(numbers) == 0:
                return 50000

            # convert to integer list
            prices = [int(n) for n in numbers]

            # take average if range exists
            price = int(np.mean(prices))

            # safety clipping
            price = max(1000, min(price, 5000000))

            return price

        except:
            return 50000

    # LOAD DATA
    def load_raw_data(self):

        possible_paths = [
            'dataset/sociolla.csv',
            'sociolla.csv',
            'dataset/skincare_products.csv'
        ]

        for path in possible_paths:

            try:
                df = pd.read_csv(path)

                if len(df) > 100:
                    return df, path

            except FileNotFoundError:
                continue

        raise FileNotFoundError("❌ Dataset tidak ditemukan!")

    # PRINT UTIL
    def print_separator(self, title):

        print("\n" + "=" * 55)
        print(f"  {title}")
        print("=" * 55)

    def print_before_after(self, label, before, after, unit="produk"):

        removed = before - after
        pct = (removed / before * 100) if before > 0 else 0

        print(f"  Before : {before:,} {unit}")
        print(f"  After  : {after:,} {unit}")
        print(f"  Removed: {removed:,} {unit} ({pct:.1f}%)")

    # MAIN CLEANING
    def clean_data(self, df):

        df_raw = df.copy()

        # STEP 1 — LOAD DATA
        self.print_separator("STEP 1 — Load Data")

        print(f"  Produk  : {len(df):,}")
        print(f"  Kolom   : {len(df.columns)}")
        print(f"  Missing : {df.isnull().sum().sum():,} values")

        print(f"\n  Sample kolom:")
        print(list(df.columns[:5]))

        # STEP 2 — DROP MISSING VALUES
        self.print_separator("STEP 2 — Drop Missing Values")

        essential_cols = [
            'product_name',
            'brand_name',
            'average_rating'
        ]

        before = len(df)

        df = df.dropna(subset=essential_cols)

        print(f"  Kolom essential: {essential_cols}")

        self.print_before_after(
            "Missing values",
            before,
            len(df)
        )

        # STEP 3 — CLEAN BRAND
        self.print_separator("STEP 3 — Clean Brand Names")

        before_brands = df['brand_name'].nunique()

        print("  Contoh BEFORE:")

        for b in df['brand_name'].head(4):
            print(f"    {b}")

        df['brand_name'] = df['brand_name'].apply(
            self.clean_brand_name
        )

        print("\n  Contoh AFTER:")

        for b in df['brand_name'].head(4):
            print(f"    {b}")

        after_brands = df['brand_name'].nunique()

        print(f"\n  Unique brands:")
        print(f"  {before_brands} → {after_brands}")

        # STEP 4 — CLEAN CATEGORY
        self.print_separator("STEP 4 — Clean Category Names")

        before_cats = df['default_category'].nunique()

        print("  Contoh BEFORE:")

        for c in df['default_category'].head(4):
            print(f"    {c}")

        df['default_category'] = df['default_category'].apply(
            self.clean_category_name
        )

        print("\n  Contoh AFTER:")

        for c in df['default_category'].head(4):
            print(f"    {c}")

        after_cats = df['default_category'].nunique()

        print(f"\n  Unique kategori:")
        print(f"  {before_cats} → {after_cats}")

        # STEP 5 — EXTRACT PRICE
        self.print_separator("STEP 5 — Extract Price")

        if 'price_range' in df.columns:

            print("  Contoh BEFORE:")

            for p in df['price_range'].head(5):
                print(f"    {p}")

            # clean price
            df['price_numeric'] = df['price_range'].apply(
                self.extract_price
            )

            print("\n  Contoh AFTER:")

            for p in df['price_numeric'].head(5):
                print(f"    Rp{p:,}")

            print("\n  Statistik harga:")
            print(f"  Min  : Rp{df['price_numeric'].min():,}")
            print(f"  Max  : Rp{df['price_numeric'].max():,}")
            print(f"  Mean : Rp{df['price_numeric'].mean():,.0f}")

        else:

            df['price_numeric'] = 50000

            print("  Kolom price_range tidak ditemukan")
            print("  Menggunakan default Rp50.000")

        # STEP 6 — CLEAN NUMERIC
        self.print_separator("STEP 6 — Filter Rating (1–5)")

        df['total_reviews'] = pd.to_numeric(
            df['total_reviews'],
            errors='coerce'
        ).fillna(0)

        df['total_in_wishlist'] = pd.to_numeric(
            df['total_in_wishlist'],
            errors='coerce'
        ).fillna(0)

        df['average_rating'] = pd.to_numeric(
            df['average_rating'],
            errors='coerce'
        )

        before = len(df)

        invalid = df[
            (df['average_rating'] < 1) |
            (df['average_rating'] > 5)
        ]

        print(f"  Rating invalid: {len(invalid)}")

        df = df[
            (df['average_rating'] >= 1) &
            (df['average_rating'] <= 5)
        ]

        self.print_before_after(
            "Filter rating",
            before,
            len(df)
        )

        print(f"\n  Rating range:")
        print(
            f"  {df['average_rating'].min():.1f}"
            f" – "
            f"{df['average_rating'].max():.1f}"
        )

        print(
            f"  Avg rating: "
            f"{df['average_rating'].mean():.2f}"
        )

        # STEP 7 — REMOVE OUTLIERS
        self.print_separator("STEP 7 — Remove Outliers")

        before = len(df)

        outlier_rev = (
            df['total_reviews'] > 10000
        ).sum()

        print(f"  Reviews > 10.000 : {outlier_rev}")

        df = df[
            df['total_reviews'] <= 10000
        ]

        self.print_before_after(
            "Remove outliers",
            before,
            len(df)
        )

        print(
            f"\n  Max reviews: "
            f"{df['total_reviews'].max():,.0f}"
        )

        # STEP 8 — FILTER RARE
        self.print_separator("STEP 8 — Filter Rare Items")

        brand_counts = df['brand_name'].value_counts()

        category_counts = df[
            'default_category'
        ].value_counts()

        valid_brands = brand_counts[
            brand_counts >= 3
        ].index

        valid_categories = category_counts[
            category_counts >= 5
        ].index

        rare_brands = (
            len(brand_counts) - len(valid_brands)
        )

        rare_cats = (
            len(category_counts) - len(valid_categories)
        )

        print(
            f"  Rare brands (<3 produk): "
            f"{rare_brands}"
        )

        print(
            f"  Rare kategori (<5 produk): "
            f"{rare_cats}"
        )

        before = len(df)

        df = df[
            df['brand_name'].isin(valid_brands)
        ]

        df = df[
            df['default_category'].isin(valid_categories)
        ]

        self.print_before_after(
            "Filter rare",
            before,
            len(df)
        )

        # STEP 9 — REMOVE UNKNOWN
        self.print_separator("STEP 9 — Remove Unknown Brand")

        before = len(df)

        unknown_count = (
            df['brand_name'] == 'Unknown'
        ).sum()

        print(f"  Unknown brand: {unknown_count}")

        df = df[
            df['brand_name'] != 'Unknown'
        ]

        self.print_before_after(
            "Remove Unknown",
            before,
            len(df)
        )

        return df

    # SAVE DATA
    def save_cleaned_data(self, df):

        output_path = os.path.join(
            self.output_dir,
            'skincare_cleaned.csv'
        )

        df.to_csv(output_path, index=False)

        return output_path

    # RUN
    def run(self):

        df_raw, path = self.load_raw_data()

        print(
            f"\n✅ Dataset ditemukan:"
            f" {path} ({len(df_raw):,} rows)"
        )

        df_clean = self.clean_data(df_raw)

        output_path = self.save_cleaned_data(df_clean)

        # SUMMARY
        print("\n" + "=" * 55)
        print("  RINGKASAN PREPROCESSING")
        print("=" * 55)

        removed = len(df_raw) - len(df_clean)

        pct = removed / len(df_raw) * 100

        print(f"  Raw dataset   : {len(df_raw):,}")
        print(f"  Clean dataset : {len(df_clean):,}")
        print(f"  Dihapus       : {removed:,} ({pct:.1f}%)")

        print(
            f"  Brands        : "
            f"{df_clean['brand_name'].nunique()}"
        )

        print(
            f"  Kategori      : "
            f"{df_clean['default_category'].nunique()}"
        )

        print(
            f"  Avg rating    : "
            f"{df_clean['average_rating'].mean():.2f}"
        )

        print(f"\n💾 Saved: {output_path}")

        print("\n✅ Preprocessing selesai!")

        return df_clean


def main():

    cleaner = SkincareDataCleaner()

    cleaner.run()


if __name__ == "__main__":
    main()