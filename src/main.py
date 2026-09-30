import time
import urllib.request
from dataclasses import dataclass
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

TARGET = "income"
HIGH_INCOME = ">50K"
CLASS_LABELS = ["<=50K", HIGH_INCOME]
FULL_TIME_HOURS = 40
NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT_COLS = [c for c in COLS if c not in NUM_COLS and c != TARGET]
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "adult.data"
FIG_DIR = ROOT / "figures"


@dataclass(frozen=True)
class ModelResult:
    """A fitted model with the held-out data and scores needed to report on it."""

    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    accuracy: float
    report: str


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
def summarize(df: pd.DataFrame) -> dict:
    """Row, column, missing-value and duplicate counts. Pure: prints nothing."""
    return {"rows": len(df), "columns": df.shape[1], "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())}  # fmt: skip


def print_overview(df: pd.DataFrame) -> None:
    """head(), info(), describe() and the missing values per column."""
    print(df.head(), "\n")
    df.info()
    print("\n", df.describe().round(2), "\n\nmissing per column:\n", df.isna().sum(), sep="")


def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def clean_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(keep="first", maintain_order=True)


# 3. filter and group
def filter_overtime_pandas(df: pd.DataFrame, hours=FULL_TIME_HOURS) -> pd.DataFrame:
    return df[df["hours_per_week"] > hours]


def filter_overtime_polars(df: pl.DataFrame, hours=FULL_TIME_HOURS) -> pl.DataFrame:
    return df.filter(pl.col("hours_per_week") > hours)


def group_by_education_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Per education level: count, mean hours, mean capital gain, share earning >50K."""
    return (
        df.assign(high_income=df[TARGET].eq(HIGH_INCOME))
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
        (pl.col(TARGET) == HIGH_INCOME).mean().alias("share_high_income"),
    ).sort("share_high_income", descending=True)  # fmt: skip


# 4. machine learning
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = all columns but the target, y = 1 if income >50K else 0."""
    return df.drop(columns=[TARGET]), df[TARGET].eq(HIGH_INCOME).astype(int).rename("high_income")


def build_model() -> Pipeline:
    """Scale numeric columns, one-hot encode text columns, then logistic regression."""
    pre = ColumnTransformer([("num", StandardScaler(), NUM_COLS),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])  # fmt: skip
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def train_and_evaluate(df: pd.DataFrame, test_size=0.2, random_state=42) -> ModelResult:
    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = classification_report(y_te, pred, target_names=CLASS_LABELS)
    return ModelResult(model, X_te, y_te, accuracy_score(y_te, pred), report)


# 5. visualisation
def _save_figure(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_income_by_age(df: pd.DataFrame, out=FIG_DIR / "income_by_age.png") -> Path:
    """Pie of income classes + stacked age histogram by income class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[TARGET].value_counts()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#4C72B0", "#DD8452"])  # fmt: skip
    ax1.set_title("Income class distribution")
    sns.histplot(data=df, x="age", hue=TARGET, bins=30, multiple="stack", ax=ax2)
    ax2.set_title("Age distribution by income class")
    fig.tight_layout()
    return _save_figure(fig, out)


def plot_confusion_matrix(result: ModelResult, out=FIG_DIR / "confusion_matrix.png") -> Path:
    disp = ConfusionMatrixDisplay.from_estimator(
        result.model, result.X_test, result.y_test, display_labels=CLASS_LABELS, cmap="Blues"
    )
    disp.ax_.set_title("Logistic regression: confusion matrix")
    return _save_figure(disp.figure_, out)


# 7. pandas vs polars
def _elapsed_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def _best_of_ms(fn, repeats) -> float:
    """Fastest of `repeats` runs, in milliseconds. The minimum is the least noisy estimate."""
    return min(_elapsed_ms(fn) for _ in range(repeats))


def benchmark(path=DATA_FILE, scale=50, repeats=5) -> pd.DataFrame:
    """Time the same operations in both libraries; filter/group run on the data repeated `scale` times."""
    raw_pd, raw_pl = load_pandas(path), load_polars(path)
    big_pd = pd.concat([clean_pandas(raw_pd)] * scale, ignore_index=True)
    big_pl = pl.concat([clean_polars(raw_pl)] * scale)
    small, big = f"{len(raw_pd):,} rows", f"{len(big_pd):,} rows"
    ops = {
        f"read_csv ({small})": (lambda: load_pandas(path), lambda: load_polars(path)),
        f"drop duplicates ({small})": (lambda: clean_pandas(raw_pd), lambda: clean_polars(raw_pl)),
        f"filter ({big})": (lambda: filter_overtime_pandas(big_pd), lambda: filter_overtime_polars(big_pl)),
        f"group_by ({big})": (lambda: group_by_education_pandas(big_pd), lambda: group_by_education_polars(big_pl)),
    }  # fmt: skip
    rows = [(name, _best_of_ms(f_pd, repeats), _best_of_ms(f_pl, repeats)) for name, (f_pd, f_pl) in ops.items()]
    out = pd.DataFrame(rows, columns=["operation", "pandas_ms", "polars_ms"])
    out["speedup"] = out["pandas_ms"] / out["polars_ms"]
    return out.round(1)


# the pipeline, one function per stage
def explore(path: Path) -> pd.DataFrame:
    """Steps 1-3: load, inspect, clean, filter and group. Returns the cleaned frame."""
    raw = load_pandas(path)
    print_overview(raw)
    print(f"\nduplicate rows: {summarize(raw)['duplicate_rows']}")
    df = clean_pandas(raw)
    print(f"\nrows after dropping duplicates: {len(df)} (was {len(raw)})")
    print(f"people working more than {FULL_TIME_HOURS} h/week: {len(filter_overtime_pandas(df))} of {len(df)}")
    print("\nper education level (pandas):\n", group_by_education_pandas(df).round(3).to_string(index=False))
    print("\nper education level (polars):\n", group_by_education_polars(clean_polars(load_polars(path))))
    return df


def model_and_plot(df: pd.DataFrame, fig_dir: Path) -> ModelResult:
    """Steps 4-5: train and score the model, save both figures."""
    result = train_and_evaluate(df)
    print(f"\nlogistic regression accuracy: {result.accuracy:.3f}\n{result.report}")
    print("saved", plot_income_by_age(df, fig_dir / "income_by_age.png"))
    print("saved", plot_confusion_matrix(result, fig_dir / "confusion_matrix.png"))
    return result


def main(path=DATA_FILE, fig_dir=FIG_DIR) -> None:
    path = download_data(path)
    df = explore(path)
    model_and_plot(df, fig_dir)
    print("\npandas vs polars (best of 5 runs):\n", benchmark(path).to_string(index=False))


if __name__ == "__main__":
    main()
