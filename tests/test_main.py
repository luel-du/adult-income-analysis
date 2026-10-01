"""Tests for src/main.py, grouped by pipeline step.

Nothing here touches the network. `raw_file` is a 6-row sample in the exact format of the UCI
file. `big_file`is a 40-row variant so the train/test split has enough rows for the model and system tests.
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
def test_summarize_counts_and_prints_nothing(df_pd, capsys):
    assert main.summarize(df_pd) == {
        "rows": 6, "columns": 15, "missing_values": 1, "duplicate_rows": 1
    }  # fmt: skip
    assert capsys.readouterr().out == ""


def test_print_overview_prints(df_pd, capsys):
    main.print_overview(df_pd)
    assert "missing per column" in capsys.readouterr().out


def test_missing_value_report(df_pd):
    report = main.missing_value_report(df_pd)
    assert report.to_dict("records") == [{"column": "workclass", "missing_rows": 1, "share_high_income": 0.0}]
    # edge case: nothing missing gives an empty report, not an error
    assert main.missing_value_report(main.clean_pandas(df_pd)).empty


def test_top_coded_report(df_pd):
    report = main.top_coded_report(df_pd, cols=("hours_per_week",)).iloc[0]
    assert (report["max"], report["rows_at_max"], report["next_highest"]) == (50, 2, 45)


def test_clean_removes_duplicates_and_labels_missing(df_pd, df_pl):
    clean_pd, clean_pl = main.clean_pandas(df_pd), main.clean_polars(df_pl)
    assert len(clean_pd) == len(clean_pl) == 5
    assert clean_pd["workclass"].tolist() == clean_pl["workclass"].to_list()
    assert clean_pd["workclass"].eq(main.MISSING_LABEL).sum() == 1
    assert clean_pd.isna().sum().sum() == 0


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
    got = pd.DataFrame(main.group_by_education_polars(main.clean_polars(df_pl)).to_dict(as_series=False))
    pd.testing.assert_frame_equal(
        expected.sort_values("education").reset_index(drop=True),
        got.sort_values("education").reset_index(drop=True),
        check_dtype=False,
    )


# 4. machine learning
def test_prepare_features(df_pd):
    X, y = main.prepare_features(main.clean_pandas(df_pd))
    assert not {"income", "fnlwgt", "education"} & set(X.columns)  # target and the two excluded columns
    assert y.tolist() == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("name", main.MODELS)
def test_every_model_predicts_binary_labels(big_file, name):
    X, y = main.prepare_features(main.clean_pandas(main.load_pandas(big_file)))
    predictions = main.build_model(name).fit(X, y).predict(X)
    assert len(predictions) == len(X) and set(predictions) <= {0, 1}


def test_model_handles_unseen_category(df_pd):
    # edge case: a category absent from training must not crash prediction
    X, y = main.prepare_features(main.clean_pandas(df_pd))
    model = main.build_model().fit(X, y)
    assert model.predict(X.head(1).assign(native_country="Atlantis"))[0] in (0, 1)


def test_evaluate_model_predicts_every_row_once(big_file):
    df = main.clean_pandas(main.load_pandas(big_file))
    result = main.evaluate_model(df)
    assert len(result.y_pred) == len(df) == 40
    assert 0.0 <= result.scores()["accuracy"] <= 1.0


def test_recall_by_sex():
    truth = pd.Series([1, 1, 1, 0])
    sex = pd.Series(["Female", "Female", "Male", "Male"])
    scores = main.ModelResult("m", truth, pd.Series([1, 0, 1, 0]), sex).scores()
    assert (scores["recall_women"], scores["recall_men"]) == (0.5, 1.0)
    # edge case: a group with no high earners has no recall; report NaN instead of failing
    no_women_high = main.ModelResult("m", pd.Series([0, 0, 1, 0]), pd.Series([0, 0, 1, 0]), sex).scores()
    assert pd.isna(no_women_high["recall_women"])


def test_compare_models_has_one_row_per_model(big_file):
    table = main.compare_models(main.clean_pandas(main.load_pandas(big_file)))
    assert table["model"].tolist() == list(main.MODELS)
    assert {"accuracy", "precision", "recall", "f1", "recall_women", "recall_men"} <= set(table.columns)


# 5. visualisation
def test_plots_write_png_files(df_pd, tmp_path):
    scores = pd.DataFrame({"model": ["a", "b"], "recall_women": [0.5, 0.7], "recall_men": [0.6, 0.9]})
    fig1 = main.plot_income_by_education(main.clean_pandas(df_pd), tmp_path / "a.png")
    fig2 = main.plot_recall_by_sex(scores, tmp_path / "b.png")
    assert fig1.stat().st_size > 0 and fig2.stat().st_size > 0


# 7. benchmark
def test_benchmark_table(raw_file):
    table = main.benchmark(raw_file, scale=2, repeats=1)
    assert list(table.columns) == ["operation", "pandas_ms", "polars_ms", "speedup"]
    assert len(table) == 4 and (table["speedup"] > 0).all()
    assert table["operation"].str.contains("6 rows").sum() == 2  # labels come from the data


# system test: the whole pipeline end to end
def test_main_runs_end_to_end(big_file, tmp_path, capsys):
    main.main(path=big_file, fig_dir=tmp_path / "figs")
    out = capsys.readouterr().out
    assert "models, 5-fold cross-validated" in out and "pandas vs polars" in out
    assert (tmp_path / "figs" / "income_by_education.png").exists()
    assert (tmp_path / "figs" / "recall_by_sex.png").exists()
