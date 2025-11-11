import polars as pl
import pytest

import probounds.probounds as pb


df_observed = pl.DataFrame(
    {
        "trt": ([1] * 1400 + [0] * 600) * 2,
        "outcome": (
            [1] * 378
            + [0] * 1022
            + [1] * 420
            + [0] * 180
            + [1] * 980
            + [0] * 420
            + [1] * 420
            + [0] * 180
        ),
        "sex": ["Female"] * 2000 + ["Male"] * 2000,
    }
)

df_experimental = pl.DataFrame(
    {
        "trt": ([1] * 1000 + [0] * 1000) * 2,
        "outcome": (
            [1] * 489
            + [0] * 511
            + [1] * 210
            + [0] * 790
            + [1] * 490
            + [0] * 510
            + [1] * 210
            + [0] * 790
        ),
        "sex": ["Female"] * 2000 + ["Male"] * 2000,
    }
)


def test_create_probounds_crosstab():
    observed_crosstab = pb.create_probounds_crosstab(df_observed, "observational")
    experimental_crosstab = pb.create_probounds_crosstab(
        df_experimental, "experimental"
    )
    assert isinstance(observed_crosstab, pl.DataFrame)
    assert isinstance(experimental_crosstab, pl.DataFrame)
    assert observed_crosstab.columns == ["trt", "0", "1", "All"]
    assert experimental_crosstab.columns == ["trt", "0", "1", "All"]

    observed_rows = observed_crosstab.to_dicts()
    expected_observed = [
        {"trt": "0", "0": 0.09, "1": 0.21, "All": 0.3},
        {"trt": "1", "0": 0.3605, "1": 0.3395, "All": 0.7},
        {"trt": "All", "0": 0.4505, "1": 0.5495, "All": 1.0},
    ]
    for row, expected in zip(observed_rows, expected_observed, strict=True):
        assert row["trt"] == expected["trt"]
        assert row["0"] == pytest.approx(expected["0"])
        assert row["1"] == pytest.approx(expected["1"])
        assert row["All"] == pytest.approx(expected["All"])

    experimental_rows = experimental_crosstab.to_dicts()
    expected_experimental = [
        {"trt": "0", "0": 0.79, "1": 0.21, "All": 1.0},
        {"trt": "1", "0": 0.5105, "1": 0.4895, "All": 1.0},
        {"trt": "All", "0": 0.65025, "1": 0.34975, "All": 1.0},
    ]
    for row, expected in zip(experimental_rows, expected_experimental, strict=True):
        assert row["trt"] == expected["trt"]
        assert row["0"] == pytest.approx(expected["0"])
        assert row["1"] == pytest.approx(expected["1"])
        assert row["All"] == pytest.approx(expected["All"])
