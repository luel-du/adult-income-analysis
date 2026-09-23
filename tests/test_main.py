"""Tests for src/main.py, grouped by pipeline step.

Nothing here touches the network. `raw_file` is a 6-row sample in the exact format of the UCI
file (leading spaces, '?' for missing, one duplicate row, a trailing blank line). `big_file`
is a 40-row variant so the train/test split has enough rows for the model and system tests.
"""

import pandas as pd
import pytest

from src import main

# ruff: noqa: E501
RAW = """\
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, ?, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 45, United-States, <=50K
53, Private, 234721, HS-grad, 9, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, >50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 50, Cuba, >50K
28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 50, Cuba, >50K

"""


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "adult.data"
    path.write_text(RAW)
    return path


@pytest.fixture
def big_file(tmp_path):
    """40 distinct rows: the 5 unique sample rows repeated with a different age each time."""
    rows = [line for line in RAW.splitlines() if line][:5]
    lines = [str(int(r.split(",")[0]) + i) + r[r.index(",") :] for i in range(8) for r in rows]
    path = tmp_path / "adult_big.data"
    path.write_text("\n".join(lines) + "\n\n")
    return path


@pytest.fixture
def df_pd(raw_file):
    return main.load_pandas(raw_file)


@pytest.fixture
def df_pl(raw_file):
    return main.load_polars(raw_file)


# 1. data loading
def test_load_pandas_parses_raw_format(df_pd):
    assert list(df_pd.columns) == main.COLS
    assert df_pd.shape == (6, 15)
    assert df_pd["income"].iloc[0] == "<=50K"  # leading space stripped
    assert df_pd["workclass"].isna().sum() == 1  # '?' became NaN
    assert df_pd["age"].dtype.kind == "i"


def test_load_polars_matches_pandas(df_pd, df_pl):
    assert df_pl.columns == main.COLS
    assert df_pl.shape == df_pd.shape  # edge case: trailing blank line is not a row
    assert df_pl["workclass"].null_count() == 1
    assert df_pl["age"].to_list() == df_pd["age"].tolist()


def test_download_data_uses_cache(raw_file):
    # edge case: an existing file is returned as-is, the (invalid) url is never fetched
    assert main.download_data(raw_file, url="http://invalid.invalid/x") == raw_file


# 2. preprocessing
def test_inspect_pandas_counts(df_pd):
    assert main.inspect_pandas(df_pd, verbose=False) == {
        "rows": 6, "columns": 15, "missing_values": 1, "duplicate_rows": 1
    }  # fmt: skip


def test_clean_removes_duplicates_in_both_libraries(df_pd, df_pl):
    assert len(main.clean_pandas(df_pd)) == len(main.clean_polars(df_pl)) == 5


def test_clean_is_idempotent(df_pd):
    # edge case: cleaning already clean data changes nothing
    once = main.clean_pandas(df_pd)
    pd.testing.assert_frame_equal(main.clean_pandas(once), once)


# 3. filtering and grouping
def test_filter_overtime(df_pd, df_pl):
    assert len(main.filter_overtime_pandas(df_pd)) == len(main.filter_overtime_polars(df_pl)) == 3
    assert len(main.filter_overtime_pandas(df_pd, hours=49)) == 2
    assert main.filter_overtime_pandas(df_pd, hours=99).empty  # edge case: nobody qualifies


def test_group_by_education_pandas(df_pd):
    grouped = main.group_by_education_pandas(main.clean_pandas(df_pd)).set_index("education")
    assert grouped.loc["HS-grad", "count"] == 2
    assert grouped.loc["HS-grad", "share_high_income"] == pytest.approx(0.5)
    assert grouped.loc["Bachelors", "mean_capital_gain"] == pytest.approx(2174 / 3)
    assert grouped["share_high_income"].is_monotonic_decreasing


def test_group_by_education_polars_equals_pandas(df_pd, df_pl):
    expected = main.group_by_education_pandas(main.clean_pandas(df_pd))
    got = main.group_by_education_polars(main.clean_polars(df_pl)).to_pandas()
    pd.testing.assert_frame_equal(
        expected.sort_values("education").reset_index(drop=True),
        got.sort_values("education").reset_index(drop=True),
        check_dtype=False,
    )


# 4. machine learning
def test_prepare_features_encodes_target(df_pd):
    X, y = main.prepare_features(df_pd)
    assert "income" not in X.columns
    assert y.tolist() == [0, 0, 0, 1, 1, 1]


def test_model_predicts_binary_labels(df_pd):
    X, y = main.prepare_features(df_pd)
    predictions = main.build_model().fit(X, y).predict(X)
    assert len(predictions) == len(X) and set(predictions) <= {0, 1}


def test_train_and_evaluate_reports_accuracy(big_file):
    result = main.train_and_evaluate(main.load_pandas(big_file))
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["X_test"]) == 8  # 20 % of 40 rows
    assert "precision" in result["report"]


def test_model_handles_unseen_category(df_pd):
    # edge case: a category absent from training must not crash prediction
    X, y = main.prepare_features(df_pd)
    model = main.build_model().fit(X, y)
    unseen = X.head(1).assign(native_country="Atlantis")
    assert model.predict(unseen)[0] in (0, 1)


# 5. visualisation
def test_plots_write_png_files(big_file, tmp_path):
    df = main.load_pandas(big_file)
    result = main.train_and_evaluate(df)
    fig1 = main.plot_income_by_age(df, tmp_path / "a.png")
    fig2 = main.plot_confusion_matrix(result["model"], result["X_test"], result["y_test"], tmp_path / "b.png")
    assert fig1.stat().st_size > 0 and fig2.stat().st_size > 0


# 7. benchmark
def test_benchmark_table(raw_file):
    table = main.benchmark(raw_file, scale=2, repeats=1)
    assert list(table.columns) == ["operation", "pandas_ms", "polars_ms", "speedup"]
    assert len(table) == 4 and (table["speedup"] > 0).all()


# system test: the whole pipeline end to end
def test_main_runs_end_to_end(big_file, tmp_path, capsys):
    main.main(path=big_file, fig_dir=tmp_path / "figs")
    out = capsys.readouterr().out
    assert "logistic regression accuracy:" in out and "pandas vs polars" in out
    assert (tmp_path / "figs" / "income_by_age.png").exists()
    assert (tmp_path / "figs" / "confusion_matrix.png").exists()
