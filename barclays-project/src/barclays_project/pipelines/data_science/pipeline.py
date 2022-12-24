"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""
from typing import List

from kedro.pipeline import Pipeline, node, pipeline

from barclays_project.pipelines.data_science import nodes as nodes
from barclays_project.pipelines.data_science.nodes import (
    collect_evaluation_metrics,
    evaluate,
    export_subselected_datasets,
    gen_optimization_study_plot,
    generate_confusion_matrix_plot,
    generate_feature_importance,
    predict_from_baseline_model_on_test_set,
    subselect_features,
    train_baseline_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                subselect_features,
                inputs=["X_train", "y_train", "categorical_cols"],
                outputs=[
                    "train_dataset",
                    "train_dataset_small",
                ],
                name="subselect_features_train",
                tags="subselect",
            ),
            node(
                subselect_features,
                inputs=["X_val", "y_val", "categorical_cols"],
                outputs=[
                    "val_dataset",
                    "val_dataset_small",
                ],
                name="subselect_features_val",
                tags="subselect",
            ),
            node(
                subselect_features,
                inputs=["X_test", "y_test", "categorical_cols"],
                outputs=[
                    "test_dataset",
                    "test_dataset_small",
                ],
                name="subselect_features_test",
                tags="subselect",
            ),
            node(
                export_subselected_datasets,
                inputs=[
                    "train_dataset",
                    "val_dataset",
                    "test_dataset",
                    "categorical_to_int_mappings",
                ],
                outputs=[
                    "train_x_csv",
                    "train_y_csv",
                    "val_x_csv",
                    "val_y_csv",
                    "test_x_csv",
                    "cat_cols_exported",
                    "cat_to_int_mappings_exported",
                ],
                name="export_subselected_datasets",
                tags="subselect",
            ),
            node(
                export_subselected_datasets,
                inputs=[
                    "train_dataset_small",
                    "val_dataset_small",
                    "test_dataset_small",
                    "categorical_to_int_mappings",
                ],
                outputs=[
                    "train_x_csv_small",
                    "train_y_csv_small",
                    "val_x_csv_small",
                    "val_y_csv_small",
                    "test_x_csv_small",
                    "cat_cols_exported_small",
                    "cat_to_int_mappings_exported_small",
                ],
                name="export_subselected_datasets_small",
                tags="subselect",
            ),
            *_create_training_pipeline("svm", "_small"),
            *_create_training_pipeline("lgbm", "_small"),
            *_create_training_pipeline("lgbm", ""),
            *_gen_feature_importance("lgbm", "_small"),
            *_gen_feature_importance("lgbm", ""),
            node(
                train_baseline_model,
                inputs=[
                    "train_dataset_small",
                ],
                outputs="baseline_model_small",
                name="train_baseline_model_small",
                tags="train_baseline_model",
            ),
            node(
                predict_from_baseline_model_on_test_set,
                inputs=["test_dataset_small", "baseline_model_small"],
                outputs="predictions_baseline_model_small",
                name="predict_from_baseline_model_on_test_set_small",
                tags="train_baseline_model",
            ),
            node(
                evaluate,
                inputs=["test_dataset_small", "predictions_baseline_model_small"],
                outputs=[
                    "evaluation_metrics_baseline_model_small",
                    "confusion_matrix_baseline_model_small",
                ],
                name="evaluation_metrics_node_baseline_model_small",
                tags=["train_baseline_model", "evaluation_metrics_nodes"],
            ),
            node(
                evaluate,
                inputs=["test_dataset_small", "predictions_ordinal_regression_model_small"],
                outputs=[
                    "evaluation_metrics_ordinal_regression_model_small",
                    "confusion_matrix_ordinal_regression_model_small",
                ],
                name="evaluation_metrics_node_ordinal_regression_model_small",
                tags=["train_ordinal_regression_model_small", "evaluation_metrics_nodes"],
            ),
            *_gen_confusion_matrix_plot("svm", "_small"),
            *_gen_confusion_matrix_plot("lgbm", "_small"),
            *_gen_confusion_matrix_plot("lgbm", ""),
            *_gen_confusion_matrix_plot("ordinal_regression_model", "_small"),
            *_gen_confusion_matrix_plot("baseline_model", "_small"),
            *_gen_optimization_study_plot("lgbm", "_small"),
            *_gen_optimization_study_plot("lgbm", ""),
            node(
                collect_evaluation_metrics,
                inputs={
                    f"evaluation_metrics_{model_name}{model_suffix}": f"evaluation_metrics_{model_name}{model_suffix}"  # noqa: E501
                    for (model_name, model_suffix) in [
                        ("svm", "_small"),
                        ("lgbm", "_small"),
                        ("lgbm", ""),
                        ("ordinal_regression_model", "_small"),
                        ("baseline_model", "_small"),
                    ]
                },
                outputs={
                    "evaluation_metrics_svm_small_df": "evaluation_metrics_svm_small_df",
                    "evaluation_metrics_svm_small_scalars": "evaluation_metrics_svm_small_scalars",
                    "evaluation_metrics_lgbm_small_df": "evaluation_metrics_lgbm_small_df",
                    "evaluation_metrics_lgbm_small_scalars": "evaluation_metrics_lgbm_small_scalars",  # noqa: E501
                    "evaluation_metrics_lgbm_df": "evaluation_metrics_lgbm_df",
                    "evaluation_metrics_lgbm_scalars": "evaluation_metrics_lgbm_scalars",
                    "evaluation_metrics_ordinal_regression_model_small_df": "evaluation_metrics_ordinal_regression_model_small_df",  # noqa: E501
                    "evaluation_metrics_ordinal_regression_model_small_scalars": "evaluation_metrics_ordinal_regression_model_small_scalars",  # noqa: E501
                    "evaluation_metrics_baseline_model_small_df": "evaluation_metrics_baseline_model_small_df",  # noqa: E501
                    "evaluation_metrics_baseline_model_small_scalars": "evaluation_metrics_baseline_model_small_scalars",  # noqa: E501
                },
                name="evaluation_metrics_report",
                tags=["evaluation_metrics_report", "evaluation_metrics_nodes"],
            ),
        ],
        namespace="data_science",
        inputs={
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
            "categorical_cols",
            "categorical_to_int_mappings",
        },
    )


def _create_training_pipeline(model_name: str, model_suffix: str) -> List[node]:
    return [
        node(
            getattr(nodes, f"hyperparam_opt_{model_name}"),
            inputs=[
                f"train_dataset{model_suffix}",
                f"val_dataset{model_suffix}",
            ],
            outputs=f"{model_name}_model{model_suffix}_study",
            name=f"hyperparam_opt_{model_name}{model_suffix}",
            tags=f"{model_name}{model_suffix}",
        ),
        node(
            getattr(nodes, f"train_final_{model_name}_model"),
            inputs=[
                f"train_dataset{model_suffix}",
                f"val_dataset{model_suffix}",
                f"{model_name}_model{model_suffix}_study",
            ],
            outputs=f"{model_name}_model{model_suffix}"
            if model_name != "svm"
            else [
                f"{model_name}_model{model_suffix}",
                f"{model_name}_standardscaler{model_suffix}",
                f"{model_name}_oh{model_suffix}",
            ],
            name=f"train_final_{model_name}_model{model_suffix}",
            tags=f"{model_name}{model_suffix}",
        ),
        node(
            getattr(nodes, f"predict_from_{model_name}_model_on_test_set"),
            inputs=[f"test_dataset{model_suffix}", f"{model_name}_model{model_suffix}"]
            + (
                []
                if model_name != "svm"
                else [
                    f"{model_name}_standardscaler{model_suffix}",
                    f"{model_name}_oh{model_suffix}",
                ]
            ),
            outputs=f"predictions_{model_name}{model_suffix}",
            name=f"predict_from_model_on_test_set_{model_name}{model_suffix}",
            tags=f"{model_name}{model_suffix}",
        ),
        node(
            evaluate,
            inputs=[f"test_dataset{model_suffix}", f"predictions_{model_name}{model_suffix}"],
            outputs=[
                f"evaluation_metrics_{model_name}{model_suffix}",
                f"confusion_matrix_{model_name}{model_suffix}",
            ],
            name=f"evaluation_metrics_node_{model_name}{model_suffix}",
            tags=[f"{model_name}{model_suffix}", "evaluation_metrics_nodes"],
        ),
    ]


def _gen_feature_importance(model_name: str, model_suffix: str) -> List[node]:
    return [
        node(
            generate_feature_importance,
            inputs=[
                f"{model_name}_model{model_suffix}",
                f"evaluation_metrics_{model_name}{model_suffix}",
            ],
            outputs=f"feature_importance_{model_name}{model_suffix}",
            name=f"generate_feature_importance_{model_name}{model_suffix}",
            tags=["generate_feature_importance_plots", "evaluation_metrics_nodes"],
        )
    ]


def _gen_confusion_matrix_plot(model_name: str, model_suffix: str) -> List[node]:
    return [
        node(
            generate_confusion_matrix_plot,
            inputs=[
                f"confusion_matrix_{model_name}{model_suffix}",
            ],
            outputs=f"confusion_matrix_plot_{model_name}{model_suffix}",
            name=f"generate_confusion_matrix_plot_{model_name}{model_suffix}",
            tags=["generate_confusion_matrix_plots", "evaluation_metrics_nodes"],
        )
    ]


def _gen_optimization_study_plot(model_name: str, model_suffix: str) -> List[node]:
    return [
        node(
            gen_optimization_study_plot,
            inputs=[
                f"{model_name}_model{model_suffix}_study",
            ],
            outputs=f"gen_optimization_study_plot_{model_name}{model_suffix}",
            name=f"gen_optimization_study_plot_{model_name}{model_suffix}",
            tags="gen_optimization_study_plots",
        )
    ]
