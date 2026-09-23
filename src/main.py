import time
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")  # headless backend so figures save in Docker and CI

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"]  # fmt: skip

NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT_COLS = [c for c in COLS if c not in NUM_COLS and c != "income"]
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "adult.data"
FIG_DIR = ROOT / "figures"


# 1. import
def download_data(path=DATA_FILE, url=URL) -> Path:
    """Download the raw CSV once and cache it on disk."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_pandas(path=DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLS, na_values="?", skipinitialspace=True)


def load_polars(path=DATA_FILE) -> pl.DataFrame:
    """Polars keeps the space after each comma and reads the trailing blank line as a row,
    so: read as text, strip, '?' -> null, cast numbers, drop all-null rows."""
    return (
        pl.read_csv(path, has_header=False, new_columns=COLS, infer_schema=False)
        .with_columns(pl.col(pl.String).str.strip_chars().replace("?", None))
        .with_columns(pl.col(NUM_COLS).cast(pl.Int64))
        .filter(~pl.all_horizontal(pl.all().is_null()))
    )


# 2. inspect and clean
def inspect_pandas(df: pd.DataFrame, verbose=True) -> dict:
    summary = {"rows": len(df), "columns": df.shape[1], "missing_values": int(df.isna().sum().sum()),
               "duplicate_rows": int(df.duplicated().sum())}  # fmt: skip
    if verbose:
        print(df.head(), "\n")
        df.info()
        print("\n", df.describe().round(2), "\n\nmissing per column:\n", df.isna().sum(), sep="")
        print(f"\nduplicate rows: {summary['duplicate_rows']}")
    return summary


def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def clean_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(keep="first", maintain_order=True)


# 3. filter and group
def filter_overtime_pandas(df: pd.DataFrame, hours=40) -> pd.DataFrame:
    return df[df["hours_per_week"] > hours]


def filter_overtime_polars(df: pl.DataFrame, hours=40) -> pl.DataFrame:
    return df.filter(pl.col("hours_per_week") > hours)


def group_by_education_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Per education level: count, mean hours, mean capital gain, share earning >50K."""
    return (
        df.assign(high_income=df["income"].eq(">50K"))
        .groupby("education")
        .agg(count=("age", "size"), mean_hours_per_week=("hours_per_week", "mean"),
             mean_capital_gain=("capital_gain", "mean"), share_high_income=("high_income", "mean"))
        .sort_values("share_high_income", ascending=False)
        .reset_index()
    )  # fmt: skip


def group_by_education_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("education").agg(
        pl.len().alias("count"),
        pl.col("hours_per_week").mean().alias("mean_hours_per_week"),
        pl.col("capital_gain").mean().alias("mean_capital_gain"),
        (pl.col("income") == ">50K").mean().alias("share_high_income"),
    ).sort("share_high_income", descending=True)  # fmt: skip


# 4. machine learning
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = all columns but income, y = 1 if income >50K else 0."""
    return df.drop(columns=["income"]), df["income"].eq(">50K").astype(int).rename("high_income")


def build_model() -> Pipeline:
    """Scale numeric columns, one-hot encode text columns, then logistic regression."""
    pre = ColumnTransformer([("num", StandardScaler(), NUM_COLS),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])  # fmt: skip
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def train_and_evaluate(df: pd.DataFrame, test_size=0.2, random_state=42) -> dict:
    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {"model": model, "X_test": X_te, "y_test": y_te, "accuracy": accuracy_score(y_te, pred),
            "report": classification_report(y_te, pred, target_names=["<=50K", ">50K"])}  # fmt: skip


# 5. visualisation
def plot_income_by_age(df: pd.DataFrame, out=FIG_DIR / "income_by_age.png") -> Path:
    """Pie of income classes + stacked age histogram by income class."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["income"].value_counts()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#4C72B0", "#DD8452"])  # fmt: skip
    ax1.set_title("Income class distribution")
    sns.histplot(data=df, x="age", hue="income", bins=30, multiple="stack", ax=ax2)
    ax2.set_title("Age distribution by income class")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_confusion_matrix(model, X_test, y_test, out=FIG_DIR / "confusion_matrix.png") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = ["<=50K", ">50K"]
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels, cmap="Blues")
    disp.ax_.set_title("Logistic regression: confusion matrix")
    disp.figure_.savefig(out, dpi=120)
    plt.close(disp.figure_)
    return out


# 7. pandas vs polars
def _time_ms(fn, repeats) -> float:
    """Best-of-N wall time in milliseconds."""
    return min(_run_ms(fn) for _ in range(repeats))


def _run_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def benchmark(path=DATA_FILE, scale=50, repeats=5) -> pd.DataFrame:
    """Time the same operations in both libraries; filter/group run on the data repeated `scale` times."""
    df_pd, df_pl = clean_pandas(load_pandas(path)), clean_polars(load_polars(path))
    big_pd, big_pl = pd.concat([df_pd] * scale, ignore_index=True), pl.concat([df_pl] * scale)
    n = f"{len(big_pd):,} rows"
    ops = {
        "read_csv (32k rows)": (lambda: load_pandas(path), lambda: load_polars(path)),
        "drop duplicates (32k rows)": (lambda: clean_pandas(df_pd), lambda: clean_polars(df_pl)),
        f"filter ({n})": (lambda: filter_overtime_pandas(big_pd), lambda: filter_overtime_polars(big_pl)),
        f"group_by ({n})": (lambda: group_by_education_pandas(big_pd), lambda: group_by_education_polars(big_pl)),
    }  # fmt: skip
    rows = [(name, _time_ms(f_pd, repeats), _time_ms(f_pl, repeats)) for name, (f_pd, f_pl) in ops.items()]
    out = pd.DataFrame(rows, columns=["operation", "pandas_ms", "polars_ms"])
    out["speedup"] = out["pandas_ms"] / out["polars_ms"]
    return out.round(1)


def main(path=DATA_FILE, fig_dir=FIG_DIR) -> None:
    path = download_data(path)
    df_pd = load_pandas(path)
    summary = inspect_pandas(df_pd)
    df_pd = clean_pandas(df_pd)
    print(f"\nrows after dropping duplicates: {len(df_pd)} (was {summary['rows']})")
    print(f"people working more than 40 h/week: {len(filter_overtime_pandas(df_pd))} of {len(df_pd)}")
    print("\nper education level (pandas):\n", group_by_education_pandas(df_pd).round(3).to_string(index=False))
    print("\nper education level (polars):\n", group_by_education_polars(clean_polars(load_polars(path))))

    result = train_and_evaluate(df_pd)
    print(f"\nlogistic regression accuracy: {result['accuracy']:.3f}\n{result['report']}")
    print("saved", plot_income_by_age(df_pd, fig_dir / "income_by_age.png"))
    print("saved", plot_confusion_matrix(result["model"], result["X_test"], result["y_test"], fig_dir / "confusion_matrix.png"))
    print("\npandas vs polars (best of 5 runs):\n", benchmark(path).to_string(index=False))


if __name__ == "__main__":
    main()
