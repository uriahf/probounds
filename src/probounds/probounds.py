from __future__ import annotations

from typing import Any, Dict, Hashable, Literal, Mapping, Sequence, TypeAlias

import polars as pl

FrameLike: TypeAlias = (
    pl.DataFrame
    | Mapping[str, Sequence[Any]]
    | Sequence[Mapping[str, Any]]
    | Sequence[Sequence[Any]]
)

Datatype = Literal["observational", "experimental"]
ProbabilityBounds = Dict[str, float]


def _ensure_polars_frame(raw_data: FrameLike) -> pl.DataFrame:
    """Return ``raw_data`` as a Polars ``DataFrame``.

    Parameters
    ----------
    raw_data : FrameLike
        Tabular data in any format accepted by :class:`polars.DataFrame`,
        including mappings of columns or sequences of records.

    Returns
    -------
    polars.DataFrame
        The input data converted to a Polars data frame.
    """

    if isinstance(raw_data, pl.DataFrame):
        return raw_data

    return pl.DataFrame(raw_data)  # type: ignore[arg-type]


def _extract_crosstab_value(
    crosstab: pl.DataFrame, row_label: Hashable, column_label: Hashable
) -> float:
    """Fetch a value from the probability bounds crosstab.

    Parameters
    ----------
    crosstab : polars.DataFrame
        Crosstab returned by :func:`create_probounds_crosstab`.
    row_label : Hashable
        Label identifying the row in the ``trt`` column to extract.
    column_label : Hashable
        Name of the column whose value should be retrieved.

    Returns
    -------
    float
        The value located at the requested row and column.

    Raises
    ------
    KeyError
        Raised when the requested row or column is not present in the
        ``crosstab``.
    """

    row_key = str(row_label)
    column_key = str(column_label)

    if column_key not in crosstab.columns:
        raise KeyError(f"Column {column_key!r} not present in crosstab")

    row_match = crosstab.filter(pl.col("trt") == row_key)
    if row_match.is_empty():
        raise KeyError(f"Row {row_key!r} not present in crosstab")

    series = row_match[column_key]
    if series.is_empty():
        raise KeyError(
            f"No value available for row {row_key!r} and column {column_key!r}"
        )
    return float(series.item())


def create_probounds_crosstab(raw_data: FrameLike, datatype: Datatype) -> pl.DataFrame:
    """Generate a normalized probability bounds crosstab.

    Parameters
    ----------
    raw_data : FrameLike
        Observational or experimental data containing ``trt`` and
        ``outcome`` columns.
    datatype : {"observational", "experimental"}
        The type of study represented by ``raw_data``. Observational data are
        normalized against the total number of records, while experimental data
        are normalized within each treatment group.

    Returns
    -------
    polars.DataFrame
        Crosstab with normalized outcome probabilities and an ``All`` summary
        column.

    Raises
    ------
    ValueError
        Raised when ``datatype`` is not supported or when ``raw_data`` does not
        contain any records.
    """

    dataframe = _ensure_polars_frame(raw_data)

    if datatype == "observational":
        normalizeby = "all"
    elif datatype == "experimental":
        normalizeby = "index"
    else:
        raise ValueError("Invalid datatype. Expected 'observational' or 'experimental'.")

    if dataframe.is_empty():
        raise ValueError("raw_data must contain at least one record")

    counts = (
        dataframe.group_by(["trt", "outcome"])
        .len()
        .pivot(index="trt", on="outcome", values="len", maintain_order=True)
        .sort("trt")
    )

    counts = counts.rename(
        {column: str(column) for column in counts.columns if column != "trt"}
    )

    expected_outcomes = ["0", "1"]
    for outcome in expected_outcomes:
        if outcome not in counts.columns:
            counts = counts.with_columns(pl.lit(0.0).alias(outcome))

    value_columns_unsorted = [column for column in counts.columns if column != "trt"]
    ordered_value_columns = [
        *[column for column in expected_outcomes if column in value_columns_unsorted],
        *[column for column in value_columns_unsorted if column not in expected_outcomes],
    ]

    counts = counts.select(["trt", *ordered_value_columns])

    counts_filled = counts.select(
        pl.col("trt").cast(pl.Utf8).alias("trt"),
        *[
            pl.col(column).fill_null(0).cast(pl.Float64).alias(column)
            for column in ordered_value_columns
        ],
    )

    total_count = float(dataframe.height)

    if normalizeby == "all":
        normalized = counts_filled.select(
            "trt",
            *[
                (pl.col(column) / total_count).alias(column)
                for column in ordered_value_columns
            ],
        )
    else:
        normalized = (
            counts_filled.with_columns(
                pl.sum_horizontal(pl.all().exclude("trt")).alias("_row_total")
            )
            .select(
                "trt",
                *[
                    pl.when(pl.col("_row_total") > 0)
                    .then(pl.col(column) / pl.col("_row_total"))
                    .otherwise(0.0)
                    .alias(column)
                    for column in ordered_value_columns
                ],
                pl.col("_row_total"),
            )
            .drop("_row_total")
        )

    normalized_with_all = (
        normalized.with_columns(
            pl.sum_horizontal(pl.all().exclude("trt")).alias("All")
        ).select(["trt", *ordered_value_columns, "All"])
    )

    column_totals = counts_filled.select(ordered_value_columns).sum()
    normalized_column_totals = (
        column_totals.with_columns(
            *[
                (pl.col(column) / total_count).alias(column)
                for column in ordered_value_columns
            ],
            pl.lit(1.0).alias("All"),
        )
        .with_columns(pl.lit("All").alias("trt"))
        .select(["trt", *ordered_value_columns, "All"])
    )

    probounds_crosstab = pl.concat(
        [normalized_with_all, normalized_column_totals], how="vertical"
    )

    ordered_columns = ["trt", *ordered_value_columns, "All"]
    return probounds_crosstab.select(ordered_columns)


def probounds_crosstab_feature(
    raw_data: FrameLike, datatype: Datatype, feature: str
) -> Dict[Hashable, pl.DataFrame]:
    """Build crosstabs for every category of ``feature``.

    Parameters
    ----------
    raw_data : FrameLike
        Input dataset that includes the ``feature`` column alongside ``trt``
        and ``outcome``.
    datatype : {"observational", "experimental"}
        The type of study represented by ``raw_data``.
    feature : str
        Column used to split the dataset prior to computing crosstabs.

    Returns
    -------
    dict[Hashable, polars.DataFrame]
        Mapping from each feature value to its probability bounds crosstab.
    """

    dataframe = _ensure_polars_frame(raw_data)

    results: Dict[Hashable, pl.DataFrame] = {}
    for group in dataframe.partition_by(feature, maintain_order=True):
        value = group[feature][0]
        results[value] = create_probounds_crosstab(group, datatype)
    return results


def calculate_bounds_observed_from_probounds_data(
    probounds_data: pl.DataFrame,
) -> ProbabilityBounds:
    """Compute bounds for observational data from a crosstab.

    Parameters
    ----------
    probounds_data : polars.DataFrame
        Crosstab generated by :func:`create_probounds_crosstab` for
        observational data.

    Returns
    -------
    dict[str, float]
        Lower and upper bounds of the benefit.
    """

    bounds_dict = {
        "lower_bound": 0.0,
        "upper_bound": _extract_crosstab_value(probounds_data, 1, 1)
        + _extract_crosstab_value(probounds_data, 0, 0),
    }
    print(
        f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}"
    )
    return bounds_dict


def calculate_bounds_experimental_from_probounds_data(
    probounds_data: pl.DataFrame,
) -> ProbabilityBounds:
    """Compute bounds for experimental data from a crosstab.

    Parameters
    ----------
    probounds_data : polars.DataFrame
        Crosstab generated by :func:`create_probounds_crosstab` for
        experimental data.

    Returns
    -------
    dict[str, float]
        Lower and upper bounds of the benefit.
    """

    bounds_dict = {
        "lower_bound": max(
            0.0,
            _extract_crosstab_value(probounds_data, 1, 1)
            - _extract_crosstab_value(probounds_data, 0, 1),
        ),
        "upper_bound": min(
            _extract_crosstab_value(probounds_data, 1, 1),
            _extract_crosstab_value(probounds_data, 0, 0),
        ),
    }
    print(
        f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}"
    )
    return bounds_dict


def calculate_bounds_combined(
    raw_data_observational: FrameLike, raw_data_experimental: FrameLike
) -> ProbabilityBounds:
    """Combine observational and experimental datasets to compute bounds.

    Parameters
    ----------
    raw_data_observational : FrameLike
        Observational dataset containing ``trt`` and ``outcome`` columns.
    raw_data_experimental : FrameLike
        Experimental dataset containing ``trt`` and ``outcome`` columns.

    Returns
    -------
    dict[str, float]
        Combined lower and upper probability bounds.
    """

    probounds_crosstab_observed = create_probounds_crosstab(
        raw_data_observational, "observational"
    )
    probounds_crosstab_experimental = create_probounds_crosstab(
        raw_data_experimental, "experimental"
    )

    prevalence = _extract_crosstab_value(probounds_crosstab_observed, "All", 1)

    lower_bound = max(
        0.0,
        prevalence - _extract_crosstab_value(probounds_crosstab_experimental, 0, 1),
        _extract_crosstab_value(probounds_crosstab_experimental, 1, 1) - prevalence,
        _extract_crosstab_value(probounds_crosstab_experimental, 1, 1)
        - _extract_crosstab_value(probounds_crosstab_experimental, 0, 1),
    )
    upper_bound = min(
        _extract_crosstab_value(probounds_crosstab_experimental, 1, 1),
        _extract_crosstab_value(probounds_crosstab_experimental, 0, 0),
        _extract_crosstab_value(probounds_crosstab_observed, 1, 1)
        + _extract_crosstab_value(probounds_crosstab_observed, 0, 0),
        _extract_crosstab_value(probounds_crosstab_experimental, 1, 1)
        - _extract_crosstab_value(probounds_crosstab_experimental, 0, 1)
        + _extract_crosstab_value(probounds_crosstab_observed, 1, 1)
        + _extract_crosstab_value(probounds_crosstab_observed, 0, 0),
    )

    bounds_dict: ProbabilityBounds = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    return bounds_dict


def calculate_bounds_combined_by_feature(
    df_observed: FrameLike, df_experimental: FrameLike, feature: str
) -> Dict[Hashable, ProbabilityBounds]:
    """Compute combined bounds for each category of ``feature``.

    Parameters
    ----------
    df_observed : FrameLike
        Observational dataset containing the ``feature`` column.
    df_experimental : FrameLike
        Experimental dataset containing the ``feature`` column.
    feature : str
        Column on which to partition the data prior to computing bounds.

    Returns
    -------
    dict[Hashable, dict[str, float]]
        Mapping of feature value to the combined probability bounds.
    """

    df_observed_polars = _ensure_polars_frame(df_observed)
    df_experimental_polars = _ensure_polars_frame(df_experimental)

    bounds_combined_by_feature: Dict[Hashable, ProbabilityBounds] = {}
    unique_values = df_observed_polars.get_column(feature).unique().to_list()

    for value in unique_values:
        filtered_observed = df_observed_polars.filter(pl.col(feature) == value)
        filtered_experimental = df_experimental_polars.filter(pl.col(feature) == value)
        bounds = calculate_bounds_combined(filtered_observed, filtered_experimental)
        bounds_combined_by_feature[value] = bounds
        print(
            f"Benefit Bounds: {bounds_combined_by_feature[value]['lower_bound']} <= Benefit|{value} <= {bounds_combined_by_feature[value]['upper_bound']}"
        )

    return bounds_combined_by_feature


def calculate_bounds_observed(df_observed: FrameLike) -> ProbabilityBounds:
    """Calculate bounds directly from observational data.

    Parameters
    ----------
    df_observed : FrameLike
        Observational dataset containing ``trt`` and ``outcome`` columns.

    Returns
    -------
    dict[str, float]
        Lower and upper bounds of the benefit for the observational dataset.
    """

    probounds_crosstab_observed = create_probounds_crosstab(
        df_observed, "observational"
    )
    return calculate_bounds_observed_from_probounds_data(probounds_crosstab_observed)


def calculate_bounds_experimental(df_experimental: FrameLike) -> ProbabilityBounds:
    """Calculate bounds directly from experimental data.

    Parameters
    ----------
    df_experimental : FrameLike
        Experimental dataset containing ``trt`` and ``outcome`` columns.

    Returns
    -------
    dict[str, float]
        Lower and upper bounds of the benefit for the experimental dataset.
    """

    probounds_crosstab_experimental = create_probounds_crosstab(
        df_experimental, "experimental"
    )
    return calculate_bounds_experimental_from_probounds_data(
        probounds_crosstab_experimental
    )


def calculate_bounds_observed_by_feature(
    df_observed: FrameLike, feature: str
) -> Dict[Hashable, ProbabilityBounds]:
    """Calculate observational bounds for each category of ``feature``.

    Parameters
    ----------
    df_observed : FrameLike
        Observational dataset containing the ``feature`` column.
    feature : str
        Column on which to partition the data prior to computing bounds.

    Returns
    -------
    dict[Hashable, dict[str, float]]
        Mapping from each feature value to its observational bounds.
    """

    dataframe = _ensure_polars_frame(df_observed)

    bounds_combined_by_feature: Dict[Hashable, ProbabilityBounds] = {}
    unique_values = dataframe.get_column(feature).unique().to_list()

    for value in unique_values:
        filtered_observed = dataframe.filter(pl.col(feature) == value)
        bounds = calculate_bounds_observed(filtered_observed)
        bounds_combined_by_feature[value] = bounds
        print(
            f"Benefit Bounds: {bounds_combined_by_feature[value]['lower_bound']} <= Benefit|{value} <= {bounds_combined_by_feature[value]['upper_bound']}"
        )

    return bounds_combined_by_feature
