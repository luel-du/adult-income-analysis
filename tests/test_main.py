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


def test_load_and_clean_agree_between_pandas_and_polars(raw_file):
    df_pd, df_pl = main.load_pandas(raw_file), main.load_polars(raw_file)
    assert len(main.clean_pandas(df_pd)) == len(main.clean_polars(df_pl)) == 5  # 6 rows, 1 duplicate


def test_group_by_education(raw_file):
    grouped = main.group_by_education_pandas(main.clean_pandas(main.load_pandas(raw_file)))
    assert grouped.set_index("education").loc["HS-grad", "share_high_income"] == pytest.approx(0.5)
