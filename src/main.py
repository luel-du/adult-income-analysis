import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")  # headless backend so figures save in Docker and CI

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"]  # fmt: skip

TARGET = "income"
HIGH_INCOME = ">50K"
FULL_TIME_HOURS = 40
MISSING_LABEL = "Unknown"
NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
TEXT_COLS = [c for c in COLS if c not in NUM_COLS and c != TARGET]
# model inputs: fnlwgt is a survey weight, not a fact about the person; education repeats education_num
MODEL_NUM = [c for c in NUM_COLS if c != "fnlwgt"]
MODEL_CAT = [c for c in TEXT_COLS if c != "education"]
MODELS = {
    "logistic regression": lambda: LogisticRegression(max_iter=1000),
    "logistic regression, class-weighted": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "gradient boosting": lambda: HistGradientBoostingClassifier(random_state=42),
}
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "adult.data"
FIG_DIR = ROOT / "figures"

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"  # colorblind-safe pair + text colors
SEX_SERIES = [
    ("recall_women", "women", BLUE, -0.13),
    ("recall_men", "men", ORANGE, 0.13),
]  # column, label, color, bar shift
CHART_STYLE = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": "#e1e0d9", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": INK, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 11,
}  # fmt: skip


@dataclass(frozen=True)
class ModelResult:
    """Out-of-fold predictions of one model, next to the truth and each person's sex."""

    name: str
    y_true: pd.Series
    y_pred: pd.Series
    sex: pd.Series

    def scores(self) -> dict:
        """Accuracy; precision, recall and F1 for the >50K class; recall for women and men separately.
        Recall = the share of actual high earners the model finds."""
        true, pred, nan = self.y_true, self.y_pred, float("nan")
        women, men = self.sex == "Female", self.sex == "Male"
        return {
            "model": self.name,
            "accuracy": accuracy_score(true, pred),
            "precision": precision_score(true, pred, zero_division=0),
            "recall": recall_score(true, pred, zero_division=0),
            "f1": f1_score(true, pred, zero_division=0),
            "recall_women": recall_score(true[women], pred[women], zero_division=nan),
            "recall_men": recall_score(true[men], pred[men], zero_division=nan),
        }


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


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columns with missing values: how many rows, and the share of high earners among those rows.
    A share far from the overall one means the gaps are not random and should not be dropped."""
    high = df[TARGET].eq(HIGH_INCOME)
    rows = [(c, int(df[c].isna().sum()), high[df[c].isna()].mean()) for c in df.columns if df[c].isna().any()]
    return pd.DataFrame(rows, columns=["column", "missing_rows", "share_high_income"])


def top_coded_report(df: pd.DataFrame, cols=("age", "capital_gain", "hours_per_week")) -> pd.DataFrame:
    """Largest value per column, how many rows sit exactly on it, and the next value below it.
    Many rows on the maximum with a gap below it means the survey capped the value there."""
    rows = [(c, df[c].max(), int((df[c] == df[c].max()).sum()), df.loc[df[c] < df[c].max(), c].max()) for c in cols]
    return pd.DataFrame(rows, columns=["column", "max", "rows_at_max", "next_highest"])


def clean_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and label missing text values, rather than dropping or guessing them."""
    return df.drop_duplicates().fillna(dict.fromkeys(TEXT_COLS, MISSING_LABEL)).reset_index(drop=True)


def clean_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(keep="first", maintain_order=True).with_columns(pl.col(TEXT_COLS).fill_null(MISSING_LABEL))


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
    """X = the model input columns, y = 1 if income >50K else 0."""
    return df[MODEL_NUM + MODEL_CAT], df[TARGET].eq(HIGH_INCOME).astype(int).rename("high_income")


def build_model(name="logistic regression") -> Pipeline:
    """Scale numeric columns, one-hot encode text columns, then one of the classifiers in MODELS."""
    dense = name == "gradient boosting"  # HistGradientBoosting cannot take a sparse matrix
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    pre = ColumnTransformer([("num", StandardScaler(), MODEL_NUM), ("cat", encoder, MODEL_CAT)])
    return Pipeline([("pre", pre), ("clf", MODELS[name]())])


def evaluate_model(df: pd.DataFrame, name="logistic regression", folds=5) -> ModelResult:
    """Cross-validate: every row is predicted by a model trained on the other folds."""
    X, y = prepare_features(df)
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    pred = cross_val_predict(build_model(name), X, y, cv=cv)
    return ModelResult(name, y, pd.Series(pred, index=y.index), df["sex"])


def compare_models(df: pd.DataFrame, folds=5) -> pd.DataFrame:
    """One row of scores per model in MODELS."""
    return pd.DataFrame([evaluate_model(df, name, folds).scores() for name in MODELS])


# 5. visualisation
def _save_figure(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_income_by_education(df: pd.DataFrame, out=FIG_DIR / "income_by_education.png") -> Path:
    """Share of people earning >50K at each education level, lowest level at the bottom."""
    high = df.assign(high=df[TARGET].eq(HIGH_INCOME))
    share = high.groupby(["education_num", "education"])["high"].mean().droplevel("education_num")
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(share)), share.to_numpy(), height=0.6, color=BLUE)
        ax.set_yticks(range(len(share)), share.index)
        for level in share.index.intersection(["HS-grad", "Bachelors", "Doctorate"]):  # label a few, not all
            ax.text(share[level] + 0.01, share.index.get_loc(level), f"{share[level]:.0%}", va="center", color=MUTED)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Share of people earning more than 50K, by education level")
        return _save_figure(fig, out)


def plot_recall_by_sex(scores: pd.DataFrame, out=FIG_DIR / "recall_by_sex.png") -> Path:
    """For each model in a compare_models() table: recall for women next to recall for men."""
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for column, label, color, shift in SEX_SERIES:
            bars = ax.barh(
                [i + shift for i in range(len(scores))], scores[column], height=0.22, color=color, label=label
            )
            ax.bar_label(bars, fmt="{:.0%}", padding=4, color=MUTED)
        ax.set_yticks(range(len(scores)), scores["model"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Share of actual high earners each model finds (recall)")
        ax.legend(frameon=False, loc="lower right")
        return _save_figure(fig, out)


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
        f"clean ({small})": (lambda: clean_pandas(raw_pd), lambda: clean_polars(raw_pl)),
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
    print("\nmissing values and the share of high earners in those rows:")
    print(missing_value_report(raw).round(3).to_string(index=False))
    print("\nvalues capped by the survey:\n", top_coded_report(raw).to_string(index=False))
    df = clean_pandas(raw)
    print(f"\nrows after cleaning: {len(df)} (was {len(raw)})")
    print(f"share earning >50K: {df[TARGET].eq(HIGH_INCOME).mean():.3f}")
    print(f"people working more than {FULL_TIME_HOURS} h/week: {len(filter_overtime_pandas(df))} of {len(df)}")
    print("\nper education level (pandas):\n", group_by_education_pandas(df).round(3).to_string(index=False))
    print("\nper education level (polars):\n", group_by_education_polars(clean_polars(load_polars(path))))
    return df


def model_and_plot(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    """Steps 4-5: compare the models and save both figures. Returns the score table."""
    scores = compare_models(df)
    print("\nmodels, 5-fold cross-validated:\n", scores.round(3).to_string(index=False))
    print("saved", plot_income_by_education(df, fig_dir / "income_by_education.png"))
    print("saved", plot_recall_by_sex(scores, fig_dir / "recall_by_sex.png"))
    return scores


def main(path=DATA_FILE, fig_dir=FIG_DIR) -> None:
    path = download_data(path)
    df = explore(path)
    model_and_plot(df, fig_dir)
    print("\npandas vs polars (best of 5 runs):\n", benchmark(path).to_string(index=False))


if __name__ == "__main__":
    main()
