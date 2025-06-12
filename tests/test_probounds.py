import probounds.probounds as pb
import pandas as pd
import numpy as np

df_observed = pd.DataFrame(
    {
        "trt": np.concatenate([np.repeat(1, 1400), np.repeat(0, 600)] * 2),
        "outcome": [1] * 378
        + [0] * 1022
        + [1] * 420
        + [0] * 180
        + [1] * 980
        + [0] * 420
        + [1] * 420
        + [0] * 180,
        "sex": ["Female"] * 2000 + ["Male"] * 2000,
    }
)

df_experimental = pd.DataFrame(
    {
        "trt": np.concatenate([np.repeat(1, 1000), np.repeat(0, 1000)] * 2),
        "outcome": [1] * 489
        + [0] * 511
        + [1] * 210
        + [0] * 790
        + [1] * 490
        + [0] * 510
        + [1] * 210
        + [0] * 790,
        "sex": ["Female"] * 2000 + ["Male"] * 2000,
    }
)


def test_create_probounds_crosstab():
    observed_crosstab = pb.create_probounds_crosstab(df_observed, "observational")
    experimental_crosstab = pb.create_probounds_crosstab(
        df_experimental, "experimental"
    )
    assert isinstance(observed_crosstab, pd.DataFrame)
    assert isinstance(experimental_crosstab, pd.DataFrame)
