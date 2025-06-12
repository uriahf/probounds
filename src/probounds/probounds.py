import pandas as pd


def create_probounds_crosstab(raw_data, datatype):
    if datatype == "observational":
        normalizeby = "all"
    elif datatype == "experimental":
        normalizeby = "index"
    else:
        raise ValueError("Invalid datatype. Expected 'observed' or 'experimental'.")

    probounds_crosstab = pd.crosstab(
        raw_data["trt"], raw_data["outcome"], margins=True, normalize=normalizeby
    )
    return probounds_crosstab


def probounds_crosstab_feature(raw_data, datatype, feature):
    results = {}
    for value, group in raw_data.groupby(feature):
        results[value] = create_probounds_crosstab(group, datatype)
    return results


def calculate_bounds_observed_from_probounds_data(probounds_data):
    bounds_dict = {
        "lower_bound": 0,
        "upper_bound": probounds_data.loc[1, 1] + probounds_data.loc[0, 0],
    }
    print(
        f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}"
    )
    return bounds_dict


def calculate_bounds_experimental_from_probounds_data(probounds_data):
    bounds_dict = {
        "lower_bound": max(0, probounds_data.loc[1, 1] - probounds_data.loc[0, 1]),
        "upper_bound": min(probounds_data.loc[1, 1], probounds_data.loc[0, 0]),
    }
    print(
        f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}"
    )
    return bounds_dict


def calculate_bounds_combined(raw_data_observational, raw_data_experimental):
    probounds_crosstab_observed = create_probounds_crosstab(
        raw_data_observational, "observational"
    )
    probounds_crosstab_experimental = create_probounds_crosstab(
        raw_data_experimental, "experimental"
    )

    prevalence = probounds_crosstab_observed.loc["All", 1]

    lower_bound = max(
        0,
        prevalence - probounds_crosstab_experimental.loc[0, 1],
        probounds_crosstab_experimental.loc[1, 1] - prevalence,
        probounds_crosstab_experimental.loc[1, 1]
        - probounds_crosstab_experimental.loc[0, 1],
    )
    upper_bound = min(
        probounds_crosstab_experimental.loc[1, 1],
        probounds_crosstab_experimental.loc[0, 0],
        probounds_crosstab_observed.loc[1, 1] + probounds_crosstab_observed.loc[0, 0],
        probounds_crosstab_experimental.loc[1, 1]
        - probounds_crosstab_experimental.loc[0, 1]
        + probounds_crosstab_observed.loc[1, 1]
        + probounds_crosstab_observed.loc[0, 0],
    )

    bounds_dict = {"lower_bound": lower_bound, "upper_bound": upper_bound}

    return bounds_dict


def calculate_bounds_combined_by_feature(df_observed, df_experimental, feature):
    bounds_combined_by_feature = {}
    unique_values = df_observed[feature].unique()

    for value in unique_values:
        filtered_observed = df_observed[df_observed[feature] == value]
        filtered_experimental = df_experimental[df_experimental[feature] == value]
        bounds = calculate_bounds_combined(filtered_observed, filtered_experimental)
        bounds_combined_by_feature[value] = bounds
        print(
            f"Benefit Bounds: {bounds_combined_by_feature[value]['lower_bound']} <= Benefit|{value} <= {bounds_combined_by_feature[value]['upper_bound']}"
        )

    return bounds_combined_by_feature


def calculate_bounds_observed(df_observed):
    probounds_crosstab_observed = create_probounds_crosstab(
        df_observed, "observational"
    )
    return calculate_bounds_observed_from_probounds_data(probounds_crosstab_observed)


def calculate_bounds_experimental(df_experimental):
    probounds_crosstab_experimental = create_probounds_crosstab(
        df_experimental, "experimental"
    )
    return calculate_bounds_experimental_from_probounds_data(
        probounds_crosstab_experimental
    )


def calculate_bounds_observed_by_feature(df_observed, feature):
    bounds_combined_by_feature = {}
    unique_values = df_observed[feature].unique()

    for value in unique_values:
        filtered_observed = df_observed[df_observed[feature] == value]
        bounds = calculate_bounds_observed(filtered_observed)
        bounds_combined_by_feature[value] = bounds
        print(
            f"Benefit Bounds: {bounds_combined_by_feature[value]['lower_bound']} <= Benefit|{value} <= {bounds_combined_by_feature[value]['upper_bound']}"
        )

    return bounds_combined_by_feature
