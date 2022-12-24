"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    concat_to_final_dataset,
    downcast_dfs,
    exploratory_analysis,
    get_mapping_of_categorical_to_ints,
    get_union_of_unique_values_for_each_col,
    load_data_and_get_unique_columns,
    load_data_and_get_unique_values,
    make_last_adjustments_and_subselect_categorical_features,
    preprocess_data,
    train_test_split_node,
    unique_cols_union,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_data_and_get_unique_columns,
                inputs="raw_dataset_partitioned",
                outputs="unique_columns_by_partition",
            ),
            node(
                unique_cols_union,
                inputs="unique_columns_by_partition",
                outputs="unique_columns_in_all_partitions",
                name="unique_cols_union",
            ),
            node(
                load_data_and_get_unique_values,
                inputs=["raw_dataset_partitioned", "unique_columns_in_all_partitions"],
                outputs="unique_values_by_column_by_partition",
                name="load_data_and_get_unique_values",
            ),
            node(
                get_union_of_unique_values_for_each_col,
                inputs="unique_values_by_column_by_partition",
                outputs="unique_values_by_column",
                name="get_union_of_unique_values_for_each_col",
            ),
            node(
                preprocess_data,
                inputs="raw_dataset_partitioned",
                outputs="parsed_datasets",
                name="preprocess_data",
            ),
            node(
                downcast_dfs,
                inputs="parsed_datasets",
                outputs="downcasted_datasets",
                name="downcast_dfs",
            ),
            node(
                concat_to_final_dataset,
                inputs="downcasted_datasets",
                outputs="preprocessed_dataset_concatted",
                name="concat_to_final_dataset",
            ),
            node(
                make_last_adjustments_and_subselect_categorical_features,
                inputs="preprocessed_dataset_concatted",
                outputs=[
                    "preprocessed_dataset_concatted_subselected",
                    "categorical_cols",
                ],
                name="make_last_adjustments_and_subselect_categorical_features",
            ),
            node(
                get_mapping_of_categorical_to_ints,
                inputs="preprocessed_dataset_concatted_subselected",
                outputs=[
                    "preprocessed_dataset_concatted_subselected_mapped_to_ints",
                    "categorical_to_int_mappings",
                ],
                name="get_mapping_of_categorical_to_ints",
            ),
            node(
                train_test_split_node,
                inputs=[
                    "preprocessed_dataset_concatted_subselected_mapped_to_ints",
                    "categorical_cols",
                ],
                outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
                name="train_test_split",
            ),
            node(
                exploratory_analysis,
                inputs="preprocessed_dataset_concatted_subselected",
                outputs="None",
                name="exploratory_analysis",
            ),
        ]
    )
