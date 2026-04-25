#!/usr/bin/env python3
"""Preprocessing scaffold for text datasets.

Usage examples:
  python scripts/preprocess.py --input data/raw/input.csv --output data/processed/output.csv
  python scripts/preprocess.py --input data/raw/input.json --output data/processed/output.parquet

This script is intentionally conservative: it implements common cleaning steps
useful for Arabic and English text (trim, lower, remove diacritics, normalize)
and is meant as a starting point for Fatima and Sayf to extend.
"""
import argparse
import os
import re
from typing import Optional

import pandas as pd


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]")


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return text
    # remove diacritics
    text = ARABIC_DIACRITICS_RE.sub("", text)
    # normalize alef forms
    text = re.sub("[إأآا]", "ا", text)
    # normalize taa marbuta
    text = re.sub("ة", "ه", text)
    # normalize ya
    text = re.sub("[يى]", "ي", text)
    # remove tatweel
    text = re.sub("ـ+", "", text)
    # collapse whitespace
    text = re.sub("\s+", " ", text).strip()
    return text


def basic_clean_text(text: str, lower: bool = True) -> str:
    if not isinstance(text, str):
        return text
    t = text
    if lower:
        t = t.lower()
    # remove control characters
    t = re.sub(r"[\r\t\n]+", " ", t)
    t = t.strip()
    return t


def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".json", ".ndjson"):
        # assume JSON lines if file is large
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {ext}")


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    normalize_ar: bool = True,
    drop_duplicates: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' not found in dataframe")

    df = df.copy()
    # basic cleaning
    df[text_column] = df[text_column].apply(basic_clean_text)

    # Arabic-specific normalization
    if normalize_ar:
        df[text_column] = df[text_column].apply(normalize_arabic)

    if dropna:
        df = df.dropna(subset=[text_column])

    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_column])

    df = df.reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess text datasets")
    parser.add_argument("--input", required=True, help="Input file path (csv/json/parquet)")
    parser.add_argument("--output", required=True, help="Output file path (csv/parquet/json)")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--no-normalize-ar", dest="normalize_ar", action="store_false", help="Disable Arabic normalization")
    parser.add_argument("--no-drop-duplicates", dest="drop_duplicates", action="store_false", help="Do not drop duplicate rows")
    parser.add_argument("--no-dropna", dest="dropna", action="store_false", help="Do not drop rows with missing text")
    parser.add_argument("--sample", type=int, default=0, help="Save a small sample of N rows instead of full output")

    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    df = load_dataframe(args.input)
    print(f"Loaded {len(df):,} rows; columns: {list(df.columns)}")

    df_clean = preprocess_dataframe(
        df,
        text_column=args.text_column,
        normalize_ar=args.normalize_ar,
        drop_duplicates=args.drop_duplicates,
        dropna=args.dropna,
    )

    if args.sample and args.sample > 0:
        out_df = df_clean.sample(min(args.sample, len(df_clean)), random_state=42)
    else:
        out_df = df_clean

    print(f"Saving {len(out_df):,} rows to {args.output}")
    save_dataframe(out_df, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Simple preprocessing scaffold for text datasets.

Usage:
  python scripts/preprocess.py --input data/raw/data.csv --output data/processed/data_clean.csv --text-col text

The script supports basic cleaning and Arabic normalization.
"""
from pathlib import Path
import argparse
import re
import pandas as pd


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return text
    # remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    # remove tatweel
    text = text.replace('ـ', '')
    # normalize alef variants to bare alef
    text = re.sub(r'[إأآ]', 'ا', text)
    # normalize yeh and alifs
    text = re.sub(r'[يى]', 'ي', text)
    # trim
    return text.strip()


def load_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in ('.json', '.ndjson'):
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def preprocess(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in dataframe")
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[text_col] = df[text_col].apply(normalize_arabic)
    # drop empty rows
    df = df[df[text_col].notna() & (df[text_col] != '')]
    # drop exact duplicates
    df = df.drop_duplicates(subset=[text_col])
    return df


def save_output(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == '.csv':
        df.to_csv(out_path, index=False)
    elif out_path.suffix.lower() in ('.json', '.ndjson'):
        df.to_json(out_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported output format: {out_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess text dataset')
    parser.add_argument('--input', '-i', required=True, help='Input file (CSV or JSON lines)')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--text-col', '-c', default='text', help='Name of the text column')
    args = parser.parse_args()

    df = load_data(args.input)
    cleaned = preprocess(df, text_col=args.text_col)
    save_output(cleaned, args.output)
    print(f"Saved cleaned data to {args.output} (rows: {len(cleaned)})")


if __name__ == '__main__':
    main()
