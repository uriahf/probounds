import pandas as pd

def create_probounds_crosstab(raw_data, datatype):
    if (datatype == 'observational'):
        normalizeby = 'all'
    elif (datatype == 'experimental'):
        normalizeby = 'index'

    probounds_crosstab = pd.crosstab(
        raw_data["trt"], 
        raw_data["outcome"], 
        margins=True,
        normalize=normalizeby)
    return probounds_crosstab

def probounds_crosstab_feature(raw_data, datatype, feature):
    results = {}
    for value, group in raw_data.groupby(feature):
        results[value] = create_probounds_crosstab(group, datatype)
    return results

def calculate_bounds_observed_from_probounds_data(probounds_data):
    bounds_dict = {
        'lower_bound': 0,
        'upper_bound': probounds_data.loc[1, 1] + probounds_data.loc[0, 0]
    }
    print(f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}")
    return bounds_dict

def calculate_bounds_experimental_from_probounds_data(probounds_data):
    bounds_dict = {
        'lower_bound': max(0, probounds_data.loc[1, 1] - probounds_data.loc[0, 1] ),
        'upper_bound': min(probounds_data.loc[1, 1], probounds_data.loc[0, 0])
    }
    print(f"Benefit Bounds: {bounds_dict['lower_bound']} <= Benefit <= {bounds_dict['upper_bound']}")
    return bounds_dict

