"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""
import logging
import os
from multiprocessing import Process
from random import seed
from typing import Callable, Dict, List, Set

import numpy as np
import pandas
import pandas as pd
from pandas import CategoricalDtype
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from barclays_project.log import configure
from barclays_project.pipelines.data_preprocessing.gps_calculations import (
    calc_distance_from_gps_coords_in_km,
)
from barclays_project.pipelines.data_preprocessing.parsing import (
    _built_form_parse,
    _energy_tariff_parse,
    _get_gps_from_address,
    _glazed_area_parse,
    _glazing_parse,
    _hotwater_parse,
    _main_fuel_parse,
    _main_heat_description_parse,
    _parse_from_to_interval,
    _tenure_parse,
)

configure()
tqdm.pandas(desc="my bar!")

seed(1337)


def load_data_and_get_unique_columns(raw_dataset: Dict[str, Callable]):
    """The input of this function is a key:value dictionary where
    every key, value pair represents a .csv file present in the raw data.
    The value is a function which upon calling loads the specific file into
    memory. The lazy loading is done in this way to not load all data into memory at once,
    as the dataset is large (>30 GB) and exceeds RAM on most personal computers."""
    unique_cols = {}
    for key, dataset in raw_dataset.items():
        unique_cols[key] = lambda dataset=dataset: _get_unique_cols(dataset())
    return unique_cols


def load_data_and_get_unique_values(raw_dataset: Dict[str, Callable], unique_columns: Set[str]):
    """The input of this function is a key:value dictionary where
    every key, value pair represents a .csv file present in the raw data.
    The value is a function which upon calling loads the specific file into
    memory. The lazy loading is done in this way to not load all data into memory at once,
    as the dataset is large (>30 GB) and exceeds RAM on most personal computers."""
    unique_values_by_col = {}
    categorical_cols = [
        "WALLS_DESCRIPTION",
        "FLOOR_LEVEL",
        "FLOOR_ENERGY_EFF",
        "MAINHEATC_ENERGY_EFF",
        "MAINHEATC_ENV_EFF",
        "FLOOR_ENV_EFF",
        "ROOF_ENV_EFF",
        "ROOF_DESCRIPTION",
        "POTENTIAL_ENERGY_RATING",
        "MAINS_GAS_FLAG",
        "HEAT_LOSS_CORRIDOR",
        "TRANSACTION_TYPE",
        "NUMBER_HEATED_ROOMS",
        "WINDOWS_DESCRIPTION",
        "MECHANICAL_VENTILATION",
        "MAIN_FUEL",
        "BUILT_FORM",
        "CONSTRUCTION_AGE_BAND",
        "ROOF_ENERGY_EFF",
        "MAINHEAT_DESCRIPTION",
        "HOTWATER_DESCRIPTION",
        "GLAZED_AREA",
        "WINDOWS_ENV_EFF",
        "PROPERTY_TYPE",
        "LIGHTING_ENV_EFF",
        "LIGHTING_ENERGY_EFF",
        "EXTENSION_COUNT",
        "POSTCODE",
        "WALLS_ENV_EFF",
        "FLOOR_DESCRIPTION",
        "MAINHEAT_ENV_EFF",
        "CURRENT_ENERGY_RATING",
        "WIND_TURBINE_COUNT",
        "GLAZED_TYPE",
        "MAINHEATCONT_DESCRIPTION",
        "SECONDHEAT_DESCRIPTION",
        "ENERGY_TARIFF",
        "SOLAR_WATER_HEATING_FLAG",
        "LIGHTING_DESCRIPTION",
        "TENURE",
        "WALLS_ENERGY_EFF",
        "FLAT_TOP_STOREY",
        "HOT_WATER_ENV_EFF",
        "HOT_WATER_ENERGY_EFF",
        "MAINHEAT_ENERGY_EFF",
        "NUMBER_HABITABLE_ROOMS",
        "WINDOWS_ENERGY_EFF",
    ]
    logging.info(
        f"Getting unique values for columns {set(categorical_cols)}, "
        f"not getting values for {unique_columns.difference(set(categorical_cols))}"
    )

    for key, dataset in raw_dataset.items():
        keys = key.split("/")
        if keys[1] == "recommendations":
            continue
        dataset_name = keys[1]
        assert dataset_name == "certificates", f"Unexpected raw data file {key} {dataset_name}"
        unique_values_by_col[key] = lambda dataset=dataset: _get_unique_values(
            dataset()[categorical_cols]
        )
    return unique_values_by_col


def _get_unique_cols(df: pd.DataFrame):
    cols = {col for col in df.columns}
    return cols


def _get_unique_values(df: pd.DataFrame):
    unique_cols_values = {}
    for col in df.columns:
        unique_cols_values[col] = set(df[col].unique().tolist())
    unique_cols_values = {
        k: {x for x in v if x == x} for k, v in unique_cols_values.items()
    }  # drop nan - use the fact that nan!=nan
    return unique_cols_values


def unique_cols_union(unique_cols: Dict[str, Callable]):
    sets = [v() for v in unique_cols.values()]
    unique_cols = set.union(*sets)
    return unique_cols


def get_union_of_unique_values_for_each_col(dataset_by_region_unique_columns: Dict[str, Callable]):
    dataset_columns = {}
    for index, (key, dataset) in enumerate(dataset_by_region_unique_columns.items()):
        dataset = dataset()
        logging.info(f"Collecting {index} columns")
        for col, value in dataset.items():
            if dataset_columns.get(col) is not None:
                dataset_columns[col] = dataset_columns[col].union(value)
            else:
                dataset_columns[col] = value
    return dataset_columns


def preprocess_data(raw_dataset: Dict[str, Callable]):
    """The input of this function is a key:value dictionary where
    every key, value pair represents a .csv file present in the raw data.
    The value is a function which upon calling loads the specific file into
    memory. The lazy loading is done in this way to not load all data into memory at once,
    as the dataset is large (>30 GB) and exceeds RAM on most personal computers.

    The unique_values_by_columns is not actually needed here.
    It was used to get the unique values in the whole dataset to make parsing and cleaning easier.
    """
    preprocessed_datasets = {}
    for key, dataset in raw_dataset.items():
        preprocessed_datasets[key] = lambda dataset=dataset: _preprocess_df(  # noqa: E501
            dataset()
        )

    return preprocessed_datasets


def _get_gps_from_df(df: pd.DataFrame):
    address = df["ADDRESS"] + ";" + df["POSTTOWN"] + ";" + df["POSTCODE"]
    address = address.str.split(";", expand=True)
    address.columns = ["street", "town", "postcode"]
    address_strs = address[address.columns].agg(", ".join, axis=1)  # noqa: F841

    lat_long = address_strs.apply(_get_gps_from_address).apply(pd.Series)
    return lat_long


subselected_categorical_cols = [
    "POSTCODE",
    "CURRENT_ENERGY_RATING",
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "MAINS_GAS_FLAG",
    "GLAZED_AREA",
    "HOTWATER_DESCRIPTION",
    "HOT_WATER_ENERGY_EFF",
    "HOT_WATER_ENV_EFF",
    "WINDOWS_DESCRIPTION",
    "WINDOWS_ENERGY_EFF",
    "WINDOWS_ENV_EFF",
    "MAINHEAT_DESCRIPTION",
    "MAINHEAT_ENERGY_EFF",
    "MAINHEAT_ENV_EFF",
    "MAIN_FUEL",
    "MECHANICAL_VENTILATION",
    "LOCAL_AUTHORITY_LABEL",
    "CONSTITUENCY_LABEL",
    "POSTTOWN",
    "TENURE",
    "GLAZING",
    "ENERGY_TARIFF",
    "IS_EPC_LABEL_BEFORE_2008_INCL",
]


def _preprocess_df(df: pd.DataFrame):
    _dfs_with_preprocessed_variables = []
    df.set_index("LMK_KEY", inplace=True)
    df_inspection_date_is_not_null = ~df["INSPECTION_DATE"].isna()
    df.loc[df_inspection_date_is_not_null, "INSPECTION_DATE_YEAR"] = pd.to_datetime(
        df.loc[df_inspection_date_is_not_null, "INSPECTION_DATE"]
    ).dt.year
    df.loc[~df_inspection_date_is_not_null, "INSPECTION_DATE_YEAR"] = "UNKNOWN"
    df["IS_EPC_LABEL_BEFORE_2008_INCL"] = df["INSPECTION_DATE_YEAR"].apply(
        lambda x: "UNKNOWN" if x == "UNKNOWN" else ("TRUE" if x <= 2008 else "FALSE")
    )
    construction_age_band = df["CONSTRUCTION_AGE_BAND"]
    construction_age_band = construction_age_band[construction_age_band != "NO DATA!"]
    construction_age_band = construction_age_band.str.split(":", expand=True)
    construction_age_band_min_max = construction_age_band[1].str.split("-", expand=True)
    construction_age_band = pd.concat(
        [construction_age_band, construction_age_band_min_max], axis=1
    )
    construction_age_band.columns = ["place", "interval", "from", "to"]

    construction_age_band = construction_age_band.apply(_parse_from_to_interval, axis=1)
    construction_age_band = construction_age_band["class"].rename("BUILDING_AGE_CLASS")
    _dfs_with_preprocessed_variables.append(construction_age_band)

    df = pd.concat([df, construction_age_band], axis=1)
    df["MAINS_GAS_FLAG"] = df["MAINS_GAS_FLAG"].apply(lambda x: "Y" if x == "Y" else "N")
    _dfs_with_preprocessed_variables.append(df["MAINS_GAS_FLAG"])
    df["GLAZING"] = df["GLAZED_TYPE"].apply(_glazing_parse)
    _dfs_with_preprocessed_variables.append(df["GLAZING"])

    df["TENURE"] = df["TENURE"].apply(_tenure_parse)
    _dfs_with_preprocessed_variables.append(df["TENURE"])

    df["GLAZED_AREA"] = df["GLAZED_AREA"].apply(_glazed_area_parse)
    df["ENERGY_TARIFF"] = df["ENERGY_TARIFF"].apply(_energy_tariff_parse)
    df["HOTWATER_DESCRIPTION"] = df["HOTWATER_DESCRIPTION"].apply(_hotwater_parse)
    df["MAIN_FUEL"] = df["MAIN_FUEL"].apply(_main_fuel_parse)
    _dfs_with_preprocessed_variables.append(
        df[["GLAZED_AREA", "ENERGY_TARIFF", "HOTWATER_DESCRIPTION", "MAIN_FUEL"]]
    )
    df["MAINHEAT_DESCRIPTION"] = df["MAINHEAT_DESCRIPTION"].apply(_main_heat_description_parse)
    df["BUILT_FORM"] = df["BUILT_FORM"].apply(_built_form_parse)
    # calculate proportion of buildings that belong to class
    # where the currently evaluated row is left out
    # as we do not know its class yet
    props = {}
    for column_name in [
        "POSTCODE",
    ]:
        logging.info(f"Calculating proportion by {column_name}")
        props[column_name] = df.groupby(column_name).apply(
            _leave_one_out_by_column_proportions, column_name, "CURRENT_ENERGY_RATING"
        )
    df = pd.concat([df] + [prop for prop in props.values()], axis=1)

    columns_to_keep = [
        "POSTCODE",
        "CURRENT_ENERGY_RATING",
        "CURRENT_ENERGY_EFFICIENCY",
        "PROPERTY_TYPE",
        "BUILT_FORM",
        "ENVIRONMENT_IMPACT_CURRENT",
        "ENERGY_CONSUMPTION_CURRENT",
        "CO2_EMISSIONS_CURRENT",
        "CO2_EMISS_CURR_PER_FLOOR_AREA",
        "LIGHTING_COST_CURRENT",
        "HEATING_COST_CURRENT",
        "HOT_WATER_COST_CURRENT",
        "TOTAL_FLOOR_AREA",
        "MAINS_GAS_FLAG",
        "MAIN_HEATING_CONTROLS",
        "MULTI_GLAZE_PROPORTION",
        "GLAZED_AREA",
        "EXTENSION_COUNT",
        "NUMBER_HABITABLE_ROOMS",
        "NUMBER_HEATED_ROOMS",
        "LOW_ENERGY_LIGHTING",
        "NUMBER_OPEN_FIREPLACES",
        "HOTWATER_DESCRIPTION",
        "HOT_WATER_ENERGY_EFF",
        "HOT_WATER_ENV_EFF",
        "WINDOWS_DESCRIPTION",
        "WINDOWS_ENERGY_EFF",
        "WINDOWS_ENV_EFF",
        "MAINHEAT_DESCRIPTION",
        "MAINHEAT_ENERGY_EFF",
        "MAINHEAT_ENV_EFF",
        "MAIN_FUEL",
        "WIND_TURBINE_COUNT",
        "MECHANICAL_VENTILATION",
        "LOCAL_AUTHORITY_LABEL",
        "CONSTITUENCY_LABEL",
        "POSTTOWN",
        "TENURE",
        "BUILDING_AGE_CLASS",
        "GLAZING",
        "ENERGY_TARIFF",
        "G_prop_POSTCODE",
        "F_prop_POSTCODE",
        "E_prop_POSTCODE",
        "D_prop_POSTCODE",
        "C_prop_POSTCODE",
        "B_prop_POSTCODE",
        "A_prop_POSTCODE",
        "POSTCODE_count",
        "IS_EPC_LABEL_BEFORE_2008_INCL",
    ]
    df.loc[:, subselected_categorical_cols] = df.loc[:, subselected_categorical_cols].fillna(
        "UNKNOWN"
    )
    df = df[columns_to_keep]
    return df


def _merge_and_pivot(df: pd.DataFrame):
    _df = df[["latitude", "longitude", "postcode", "BUILDING_REFERENCE_NUMBER"]]
    _df = _df.dropna(subset="latitude")
    _df = _df.merge(_df, how="outer", on="postcode")
    # drop identical buildings
    _df = _df[_df["BUILDING_REFERENCE_NUMBER_x"] != _df["BUILDING_REFERENCE_NUMBER_y"]]
    # calculate distance
    _df["distance"] = _df.apply(calc_distance_from_gps_coords_in_km, axis=1)
    _df = _df.set_index(["BUILDING_REFERENCE_NUMBER_x", "BUILDING_REFERENCE_NUMBER_y"])
    distance = _df["distance"]
    # create distance matrix
    distance = distance.unstack()
    return distance


def _leave_one_out_by_column_proportions(df: pd.DataFrame, column: str, target: str):
    collected_rows = {}
    _necessary_cols = {"A", "B", "C", "D", "E", "F", "G"}
    logging.info(
        f"Calculating proportions of target for variable {column} {df[column].unique()[0]}"
    )
    if df.empty or len(df) == 1:
        for col in _necessary_cols:
            df[col + "_prop_" + column] = np.nan
        df[column + "_count"] = len(df)
        return df[[col + "_prop_" + column for col in _necessary_cols]]  # mock empty nans
    for row_num, (i, row) in enumerate(df.iterrows()):
        _df = df.loc[df.index != i, :]  # leave out current row
        proportions = pd.crosstab(index=_df[column], columns=_df[target], normalize="index")
        if len(set(proportions.columns).intersection(_necessary_cols)) != len(
            _necessary_cols
        ):  # if some columns are not present, fill with na
            for _col in _necessary_cols.difference(set(proportions.columns)):
                proportions[_col] = np.nan
        collected_rows[i] = proportions.T.iloc[:, 0].rename(i)
    proportions_by_target = pd.concat(collected_rows.values(), axis=1).T
    proportions_by_target.columns = [
        col + "_prop_" + column for col in proportions_by_target.columns
    ]
    proportions_by_target[column + "_count"] = len(df) - 1
    proportions_by_target = proportions_by_target.fillna(0)
    return proportions_by_target


def downcast_dfs(parsed_datasets: Dict[str, Callable]):
    dfs = {}
    for i, (key, dataset) in enumerate(parsed_datasets.items()):
        dfs[key] = lambda dataset=dataset: _downcast_df(dataset())
    return dfs


def _downcast_df(dataset: pd.DataFrame):

    for col in subselected_categorical_cols:
        dataset[col] = dataset[col].astype("category")
    fcols = dataset.select_dtypes("float").columns
    dataset[fcols] = dataset[fcols].apply(pd.to_numeric, downcast="float")
    return dataset


def _save_partitions_to_file(parsed_datasets: Dict[str, Callable], file: str):
    os.remove(file)
    with open(file, "wb") as f:
        for i, df in enumerate(parsed_datasets.values()):
            logging.info(f"Saving file {i} out of {len(parsed_datasets.values())}")
            if i == 0:
                df().to_csv(f, index=False)
            else:
                df().to_csv(f, mode="a", index=False, header=False)


def concat_to_final_dataset(parsed_datasets: Dict[str, Callable]):
    # it is necessary to do this this way since otherwise
    # concatting dataframes in memory crashes OOM
    file = "./data/02_intermediate/concatting_file.csv"
    dtypes = list(parsed_datasets.items())[0][1]().dtypes.to_dict()
    dtypes["BUILDING_AGE_CLASS"] = pandas.Float32Dtype
    p = Process(target=_save_partitions_to_file, args=(parsed_datasets, file))
    p.start()
    p.join()
    p.kill()
    dtypes = {k: "category" for k, v in dtypes.items() if isinstance(v, CategoricalDtype)}
    concatted_file = pd.read_csv(file, dtype=dtypes)
    concatted_file = concatted_file[
        concatted_file["CURRENT_ENERGY_RATING"].isin({"A", "B", "C", "D", "E", "F", "G"})
    ]
    return concatted_file


def get_mapping_of_categorical_to_ints(df: pd.DataFrame):
    category_mapping = {}
    for col in df.columns:
        if not isinstance(df[col].dtype, pandas.CategoricalDtype):
            continue
        category_mapping[col] = {}
        # target is encoded such that the levels are ordered (they are ordinal)
        if col == "CURRENT_ENERGY_RATING":
            int_to_category_mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
        else:
            int_to_category_mapping = dict(enumerate(df[col].cat.categories))
        category_to_int_mappings = {v: k for k, v in int_to_category_mapping.items()}
        category_mapping[col]["int_to_category_mapping"] = int_to_category_mapping
        category_mapping[col]["category_to_int_mappings"] = category_to_int_mappings
        df[col] = df[col].map(category_to_int_mappings)
    return df, category_mapping


# why we chose to calculate the proportions in each postcode manually
# because there are usually only a few members in a given postcode
# we need features with as few levels as possible to be able to predict
# since the model needs to see all levels in the train set


def make_last_adjustments_and_subselect_categorical_features(df: pd.DataFrame):
    # drop observations where there is only one observation in POSTCODE to enable stratification
    postcode_value_counts = df["POSTCODE"].value_counts()
    postcode_value_counts_greater_than_1 = postcode_value_counts[postcode_value_counts > 1]
    mask = df["POSTCODE"].isin(postcode_value_counts_greater_than_1.index)
    logging.info(
        f"Dropping {sum(~mask)} observations that are a single observation in given postcode"
    )
    df = df[mask]

    cols_to_drop = [
        "MAIN_HEATING_CONTROLS",
        "NUMBER_HEATED_ROOMS",
        "MULTI_GLAZE_PROPORTION",
        "LOW_ENERGY_LIGHTING",
        "EXTENSION_COUNT",
        "NUMBER_HEATED_ROOMS",
        "NUMBER_OPEN_FIREPLACES",
        "WIND_TURBINE_COUNT",
        "POSTCODE",
        "CONSTITUENCY_LABEL",
        "POSTTOWN",
    ]
    df = df.drop(cols_to_drop, axis=1)
    # the values dropped here are dropped only if there was only 1 observation
    # for given postcode - we do not want to consider such properties anyway
    # (we do not have much information about them)
    df = df.dropna(
        subset=[
            "G_prop_POSTCODE",
            "F_prop_POSTCODE",
            "E_prop_POSTCODE",
            "D_prop_POSTCODE",
            "C_prop_POSTCODE",
            "B_prop_POSTCODE",
            "A_prop_POSTCODE",
            "NUMBER_HABITABLE_ROOMS",
            "TOTAL_FLOOR_AREA",
            "BUILDING_AGE_CLASS",
        ]
    )
    # if there are atleast 3 buildings in given postcode
    df["POSTCODE_PROPORTIONS_ARE_RELIABLE_IND"] = df["POSTCODE_count"] > 5
    df["POSTCODE_PROPORTIONS_ARE_RELIABLE_IND"] = df[
        "POSTCODE_PROPORTIONS_ARE_RELIABLE_IND"
    ].astype("category")
    df = df[df["MECHANICAL_VENTILATION"] != "NO DATA!"]
    df = df[df["GLAZED_AREA"] != "UNKNOWN"]
    df = df[df["BUILDING_AGE_CLASS"] != "UNKNOWN"]
    # this is an ugly hack to make train test split work
    # without this the train test split failes, because
    # the IS_EPC_LABEL_BEFORE_2008_INCL has values TRUE/FALSE/UNKNOWN
    # and unfortunately, printing this to csv and then loading again
    # converts the TRUE and FALSE to True and False and then
    # True cannot be compared with "UNKNOWN"
    df["IS_EPC_LABEL_BEFORE_2008_INCL"] = df["IS_EPC_LABEL_BEFORE_2008_INCL"].apply(
        lambda x: "TRUE"
        if (isinstance(x, bool) and x)
        else ("FALSE" if (isinstance(x, bool) and not x) else x)
    )
    df["IS_EPC_LABEL_BEFORE_2008_INCL"] = df["IS_EPC_LABEL_BEFORE_2008_INCL"].astype("str") + "_"
    # very few observations are UNKNOWN  - drop them
    df = df[df["IS_EPC_LABEL_BEFORE_2008_INCL"] != "UNKNOWN_"]
    df["IS_EPC_LABEL_BEFORE_2008_INCL"] = df["IS_EPC_LABEL_BEFORE_2008_INCL"].astype("category")

    assert df.isna().sum().sum() == 0

    global subselected_categorical_cols
    categorical_cols = list(
        set(subselected_categorical_cols).difference(
            {
                "CURRENT_ENERGY_RATING",
                "POSTCODE",
                "WINDOWS_DESCRIPTION",
                "CONSTITUENCY_LABEL",
                "POSTTOWN",
                "BUILDING_AGE_CLASS",
            }.union(set(cols_to_drop))
        )
    )

    return df, categorical_cols


def train_test_split_node(df: pd.DataFrame, categorical_cols: List[str]):
    y = df["CURRENT_ENERGY_RATING"]
    X = df.drop(["CURRENT_ENERGY_RATING"], axis=1)
    # split into train and test set
    # we need small train and val to be able to train the model
    # since the dataset is too large

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=13373, stratify=y
    )
    for col in categorical_cols:
        assert sorted(X[col].unique()) == sorted(X_train[col].unique()), (
            f"Categorical feature {col} does not contain all levels in "
            f"train set, {sorted(X[col].unique())}, {sorted(X_train[col].unique())}"
        )

    # take only smaller val set to make training atleast a bit fast
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.33, random_state=1337, stratify=y_test
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def exploratory_analysis(df: pd.DataFrame):
    profile = ProfileReport(df, minimal=True, title="Pandas Profiling Report")
    profile.to_file("exploratory_analysis_profile_report.html")
    return 0
