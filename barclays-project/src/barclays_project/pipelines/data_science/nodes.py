import logging
from typing import Dict, List, Set

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from barclays_project.pipelines.data_science.ranked_probability_score import (
    ranked_probability_score,
)


def subselect_features(X: pd.DataFrame, y: pd.DataFrame, categorical_cols: List[str]):
    categorical_cols = set(categorical_cols)
    cols_small = [
        "PROPERTY_TYPE",
        "BUILT_FORM",
        "TOTAL_FLOOR_AREA",
        "NUMBER_HABITABLE_ROOMS",
        "BUILDING_AGE_CLASS",
        "LOCAL_AUTHORITY_LABEL",
        "G_prop_POSTCODE",
        "F_prop_POSTCODE",
        "E_prop_POSTCODE",
        "D_prop_POSTCODE",
        "C_prop_POSTCODE",
        "B_prop_POSTCODE",
        "A_prop_POSTCODE",
        "POSTCODE_count",
        "IS_EPC_LABEL_BEFORE_2008_INCL",
        "POSTCODE_PROPORTIONS_ARE_RELIABLE_IND",
    ]
    small_dataset_X = X[cols_small]
    # dropping variables that had very high impact on the prediction
    # - probably criteria for classification into
    # one of the classes?
    large_cols = set(X.columns).difference(
        {
            "CURRENT_ENERGY_EFFICIENCY",
            "ENVIRONMENT_IMPACT_CURRENT",
            "CO2_EMISS_CURR_PER_FLOOR_AREA",
            "HEATING_COST_CURRENT",
            "ENERGY_CONSUMPTION_CURRENT",
            "CO2_EMISSIONS_CURRENT",
            "HOT_WATER_COST_CURRENT",
        }
    )

    large_cols = list(large_cols)
    X = X[large_cols]
    categorical_cols_large = set(X.columns).intersection(categorical_cols)
    categorical_cols_large.add("POSTCODE_PROPORTIONS_ARE_RELIABLE_IND")
    categorical_cols_small = set(small_dataset_X.columns).intersection(categorical_cols)
    categorical_cols_small.add("POSTCODE_PROPORTIONS_ARE_RELIABLE_IND")

    return (X, y, categorical_cols_large), (small_dataset_X, y, categorical_cols_small)


def export_subselected_datasets(train, val, test, categorical_to_int_mappings):
    X_train, y_train, cat_cols_train = train
    X_val, y_val, cat_cols_val = val
    X_test, y_test, cat_cols_test = test
    assert cat_cols_train == cat_cols_val
    assert cat_cols_test == cat_cols_val
    # purposefully not returning y_test so that it cannot be used in the R script
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        list(cat_cols_train),
        categorical_to_int_mappings,
    )


def hyperparam_opt_lgbm(train_dataset, val_dataset):
    X_train, y_train, categorical_cols = train_dataset
    X_val, y_val, _ = val_dataset
    study = optuna.create_study(direction="minimize", study_name="LGBM model")
    func = lambda trial: _lgbm_objective(  # noqa: E731
        trial, X_train, y_train, X_val, y_val, categorical_cols
    )
    study.optimize(func, show_progress_bar=True, n_jobs=1, timeout=3600 * 8)
    # plot_optimization_history(study).show()
    return study


def _lgbm_objective(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    categorical_cols: Set,
):
    objective = "multiclass"
    num_class = 7
    n_estimators = 300
    early_stopping_round = 10
    # is_unbalance = True
    normalized_value_counts = y_train.value_counts(normalize=True)
    class_weight = 1 / normalized_value_counts
    class_weight = dict(zip([x[0] for x in class_weight.index], class_weight))
    # to run predefined distributino, use the below use_predefined_distribution flag
    # to run balanced distribution, use is_unbalance=True
    # to run unbalanced distribution, use is_unbalance=False without class weight
    use_predefined_distribution = True
    if use_predefined_distribution:
        normalized_value_counts.index = [x[0] for x in normalized_value_counts.index]
        class_weight_predefined_distribution = pd.Series(
            {0.0: 0.003, 1.0: 0.163, 2.0: 0.227, 3.0: 0.389, 4.0: 0.172, 5.0: 0.039, 6.0: 0.007}
        )
        class_weight = class_weight_predefined_distribution / normalized_value_counts
        class_weight = dict(zip([x for x in class_weight.index], class_weight))
    categorical_feature = f"name:{','.join(list(categorical_cols))}"
    trial.set_user_attr("objective", objective)
    trial.set_user_attr("num_class", num_class)
    trial.set_user_attr("n_estimators", n_estimators)
    trial.set_user_attr("early_stopping_round", early_stopping_round)
    trial.set_user_attr("categorical_feature", categorical_feature)
    # trial.set_user_attr("is_unbalance", is_unbalance)
    trial.set_user_attr("class_weight", class_weight)

    lgbm_params = {
        "objective": objective,
        "num_class": num_class,
        "n_estimators": n_estimators,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        "early_stopping_round": early_stopping_round,
        "categorical_feature": categorical_feature,
        "class_weight": class_weight,
    }
    model = train_lgbm_model(X_train, y_train, X_val, y_val, lgbm_params)
    valid_loss = model.best_score_["valid_0"]["multi_logloss"]
    return valid_loss


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: Dict,
):
    model = lgb.LGBMClassifier(**params)
    return model.fit(X_train, y_train, eval_set=(X_val, y_val))


def train_final_lgbm_model(train_dataset, val_dataset, study):
    X_train, y_train, categorical_cols = train_dataset
    X_val, y_val, _ = val_dataset
    params = study.best_params
    params["categorical_feature"] = f"name:{','.join(list(categorical_cols))}"
    params = {**params, **study.best_trial.user_attrs}
    params["n_estimators"] = 1000
    model = train_lgbm_model(X_train, y_train, X_val, y_val, params)
    return model


def hyperparam_opt_svm(train_dataset, val_dataset):
    X_train, y_train, categorical_cols = train_dataset
    X_val, y_val, _ = val_dataset

    categorical_cols = categorical_cols.difference({"LOCAL_AUTHORITY_LABEL"})
    oh = OneHotEncoder()
    X_train_transformed = oh.fit_transform(X_train[categorical_cols])
    ohe_df = pd.DataFrame(X_train_transformed.toarray(), columns=oh.get_feature_names_out())
    X_train = pd.concat([X_train, ohe_df], axis=1).drop(categorical_cols, axis=1)
    X_val_transformed = oh.transform(X_val[categorical_cols])
    ohe_df_val = pd.DataFrame(X_val_transformed.toarray(), columns=oh.get_feature_names_out())
    X_val = pd.concat([X_val, ohe_df_val], axis=1).drop(categorical_cols, axis=1)

    for col in X_train.columns:
        if (col in oh.get_feature_names_out()) or (
            isinstance(X_train[col].dtype, pd.CategoricalDtype)
        ):
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    study = optuna.create_study(direction="minimize", study_name="SVM model")
    func = lambda trial: _svm_objective(  # noqa: E731
        trial, X_train, y_train, X_val, y_val, categorical_cols, oh
    )
    study.optimize(func, show_progress_bar=True, n_jobs=1, timeout=8 * 3600)
    return study


def _svm_objective(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    categorical_cols: Set,
    onehot_encoder: OneHotEncoder,
):
    random_state = 1
    verbose = 1
    n_jobs = 6
    # modified_huber loss is equivalent to a quadratically smoothed SVM with gamma = 2
    # https://datascience.stackexchange.com/questions/20217/which-algorithm-
    # is-used-in-sklearn-sgdclassifier-when-modified-huber-loss-is-use
    # https://www.quora.com/What-algorithm-is-used-in-sklearn%E2%80%99s
    # -SGDClassifier-when-a-modified-huber-loss-is-used/
    loss = "modified_huber"
    # class_weight = "balanced"
    alpha = trial.suggest_float("alpha", 0.001, 0.99)
    early_stopping = True
    n_iter_no_change = 50
    trial.set_user_attr("random_state", random_state)
    trial.set_user_attr("verbose", verbose)
    # use the class_weight = "balanced" for balanced svm
    # trial.set_user_attr("class_weight", class_weight)
    trial.set_user_attr("loss", loss)
    trial.set_user_attr("n_jobs", n_jobs)
    trial.set_user_attr("categorical_cols", list(categorical_cols))
    trial.set_user_attr("onehot_encoder", onehot_encoder)
    trial.set_user_attr("early_stopping", early_stopping)
    trial.set_user_attr("n_iter_no_change", n_iter_no_change)

    model = SGDClassifier(
        loss=loss,
        alpha=alpha,
        verbose=verbose,
        # class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
    )
    model.fit(X_train, y_train)
    loss = model.score(X_val, y_val)
    return loss


def train_final_svm_model(train_dataset, val_dataset, study):
    X_train, y_train, _ = train_dataset
    X_val, y_val, _ = val_dataset
    user_attrs = study.best_trial.user_attrs
    oh = user_attrs.pop("onehot_encoder")
    categorical_cols = user_attrs.pop("categorical_cols")
    X_train_transformed = oh.transform(X_train[categorical_cols])
    ohe_df = pd.DataFrame(X_train_transformed.toarray(), columns=oh.get_feature_names_out())
    X_train = pd.concat([X_train, ohe_df], axis=1).drop(categorical_cols, axis=1)
    X_val_transformed = oh.transform(X_val[categorical_cols])
    ohe_df_val = pd.DataFrame(X_val_transformed.toarray(), columns=oh.get_feature_names_out())
    X_val = pd.concat([X_val, ohe_df_val], axis=1).drop(categorical_cols, axis=1)

    for col in X_train.columns:
        if (col in oh.get_feature_names_out()) or (
            isinstance(X_train[col].dtype, pd.CategoricalDtype)
        ):
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    categorical_columns = list(X_train.select_dtypes("int64").columns)
    numerical_columns = list(set(X_train.columns).difference(set(categorical_columns)))
    standard_scaler = StandardScaler()
    X_train[numerical_columns] = standard_scaler.fit_transform(X_train[numerical_columns], y_train)
    params = study.best_params
    params = {**params, **user_attrs}
    # set params here manually
    params["n_iter_no_change"] = params["n_iter_no_change"] + 500
    model = SGDClassifier(**params)
    model.fit(X_train, y_train)
    model.oh = oh
    return model, standard_scaler, oh


def predict_from_lgbm_model_on_test_set(test_dataset, model):
    X_test, y_test, categorical_cols = test_dataset
    predictions = model.predict_proba(X_test)
    predictions = pd.DataFrame(predictions)
    predictions.columns = model.classes_
    return predictions


def predict_from_svm_model_on_test_set(test_dataset, model, standard_scaler, oh):
    X_test, y_test, categorical_cols = test_dataset
    oh = model.oh
    categorical_cols = list(oh.feature_names_in_)
    X_test_transformed = oh.transform(X_test[categorical_cols])
    ohe_df_val = pd.DataFrame(X_test_transformed.toarray(), columns=oh.get_feature_names_out())
    X_test = pd.concat([X_test, ohe_df_val], axis=1).drop(categorical_cols, axis=1)

    for col in X_test.columns:
        if isinstance(X_test[col].dtype, pd.CategoricalDtype):
            X_test[col] = X_test[col].astype(int)

    numerical_columns = list(standard_scaler.get_feature_names_out())
    X_test[numerical_columns] = standard_scaler.transform(X_test[numerical_columns])

    predictions = model.predict_proba(X_test)
    predictions = pd.DataFrame(predictions)
    predictions.columns = model.classes_
    return predictions


def train_baseline_model(train_dataset):
    X_train, y_train, categorical_cols = train_dataset
    balanced = True
    props = y_train.value_counts(normalize=True)
    if balanced:
        props = props * 0 + 1 / len(props)
        assert abs(sum(props) - 1) < 0.001, "Props for baseline model do not sum up to 1"
    props = props.sort_index()
    return props


def predict_from_baseline_model_on_test_set(test_dataset, model):
    X_test, y_test, _ = test_dataset
    predictions = pd.DataFrame(np.zeros((len(y_test), len(model))))
    for index, value in model.iteritems():
        predictions.loc[:, index] = value
    return predictions


def evaluate(dataset_containing_y_target, y_pred: pd.DataFrame):
    """

    Args:
        y_target: target label
        y_pred: probabilities of targets

    Returns: ranked probability score

    """
    _, y_target, _ = dataset_containing_y_target
    y_pred.columns = [float(col) for col in y_pred.columns]
    if np.all(
        y_pred.iloc[:2000, :].sum(axis=1) > 1
    ):  # lgbm one vs all outputs nonnormalized predictions
        y_pred = y_pred.div(y_pred.sum(axis=1), axis=0)
    predicted_classes = np.argmax(y_pred.to_numpy(), axis=1)
    class_report = classification_report(y_target, predicted_classes, output_dict=True)
    target_classes = sorted(list(y_target["CURRENT_ENERGY_RATING"].unique()))
    dummies = pd.get_dummies(y_target["CURRENT_ENERGY_RATING"])
    rps = {}
    for tar in target_classes:
        mask = y_target["CURRENT_ENERGY_RATING"] == tar
        rps[tar] = ranked_probability_score(dummies[mask], y_pred[mask])
        class_report[f"{tar}"]["ranked_probability_score"] = rps[tar]
    logging.info(f"Classification report {class_report}")
    conf_matrix = confusion_matrix(y_target, predicted_classes)
    return class_report, conf_matrix


def generate_feature_importance(model, evaluation_metrics):
    ax = lgb.plot_importance(model, height=0.2, max_num_features=60, importance_type="gain")
    fig = ax.figure
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.close()
    return fig


def generate_confusion_matrix_plot(confusion_matrix):
    target_names = ["A", "B", "C", "D", "E", "F", "G"]
    cmn = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix = sn.heatmap(
        cmn, annot=True, xticklabels=target_names, yticklabels=target_names, vmin=0, vmax=1
    )
    conf_matrix.set(xlabel="Predicted class", ylabel="Target class")
    conf_matrix_figure = conf_matrix.figure
    return conf_matrix_figure


def collect_evaluation_metrics(**kwargs):
    evaluation_metrics_dict = kwargs
    df_dict = {}
    mapping_dict = {
        "0.0": "A",
        "1.0": "B",
        "2.0": "C",
        "3.0": "D",
        "4.0": "E",
        "5.0": "F",
        "6.0": "G",
    }
    for eval_metric_name, eval_report in evaluation_metrics_dict.items():
        dicts = {
            (mapping_dict[name] if (name in mapping_dict.keys()) else name): x
            for name, x in eval_report.items()
            if isinstance(x, dict)
        }
        vals = {name: value for name, value in eval_report.items() if name not in dicts.keys()}
        df1 = pd.DataFrame.from_dict(dicts).T
        df_dict[eval_metric_name + "_df"] = df1
        df_dict[eval_metric_name + "_scalars"] = pd.DataFrame(vals, index=[0])

    latex = []
    for key, df in df_dict.items():
        latex.append(df.to_latex(caption=key))

    return df_dict


def gen_optimization_study_plot(optimization_study):
    ax = plot_optimization_history(optimization_study)
    fig = ax.figure
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.close()
    return fig
