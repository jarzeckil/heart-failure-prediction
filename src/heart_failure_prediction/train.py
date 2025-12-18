import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import mlflow
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_failure_prediction.config import PROJECT_ROOT
from heart_failure_prediction.preprocessing import ZeroImputer

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline(cfg: DictConfig) -> Pipeline:
    zero_imputer_columns = list(cfg.processing.missing_vals_cols)
    num_columns = list(cfg.processing.num_features)
    cat_columns = list(cfg.processing.cat_features)
    num_imp_strategy = cfg.processing.num_impute_strategy
    cat_imp_strategy = cfg.processing.cat_impute_strategy

    model = hydra.utils.instantiate(cfg.model.estimator)

    full_pipeline = Pipeline(
        [
            (
                'preprocessing',
                ColumnTransformer(
                    [
                        (
                            'num_pipeline',
                            Pipeline(
                                [
                                    ('zero_imputer', ZeroImputer(zero_imputer_columns)),
                                    (
                                        'median_imputer',
                                        SimpleImputer(
                                            strategy=num_imp_strategy,
                                            add_indicator=True,
                                        ),
                                    ),
                                    ('scaler', StandardScaler()),
                                ]
                            ),
                            num_columns,
                        ),
                        (
                            'cat_pipeline',
                            Pipeline(
                                [
                                    (
                                        'most_frequent_imputer',
                                        SimpleImputer(strategy=cat_imp_strategy),
                                    ),
                                    (
                                        'one_hot_encoder',
                                        OneHotEncoder(
                                            handle_unknown='ignore', drop='first'
                                        ),
                                    ),
                                ]
                            ),
                            cat_columns,
                        ),
                    ]
                ),
            ),
            ('model', model),
        ]
    )

    return full_pipeline


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(
        f'Evaluation metrics: '
        f'accuracy = {accuracy} '
        f'recall = {recall} '
        f'precision = {precision} '
        f'f1 = {f1}'
    )

    scores = {
        'accuracy': float(accuracy),
        'recall': float(recall),
        'precision': float(precision),
        'f1_score': float(f1),
    }

    return scores


def split_data(cfg: DictConfig, df: pd.DataFrame) -> tuple:
    target = cfg.model.target
    test_size = cfg.model.test_size
    random_state = cfg.model.random_state

    y = df[target]
    X = df.drop(target, axis=1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def log_feature_importance(pipeline, figsize=(14, 14)):
    preprocessor = pipeline.named_steps['preprocessing']
    feature_names = preprocessor.get_feature_names_out()

    model = pipeline.named_steps['model']
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        print('Error: dimensions are not right.')
        return

    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_imp_df,
        palette='viridis',
        hue='Feature',
        legend=False,
    )

    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature name')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(
        HydraConfig.get().runtime.output_dir, 'feature_importance.png'
    )
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
    logger.info('Feature importance plot logged to MLflow')


@hydra.main(
    config_path=os.path.join(PROJECT_ROOT, 'conf'),
    config_name='config',
    version_base='1.2',
)
def main(cfg: DictConfig) -> float:
    mlflow.set_experiment('Heart failure prediction')

    data = load_data(path=hydra.utils.to_absolute_path(cfg.raw_data.path))
    X_train, X_test, y_train, y_test = split_data(cfg, data)

    model = build_pipeline(cfg)
    model_class_name = model.named_steps['model'].__class__.__name__

    run_name = f'{model_class_name}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.processing)
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('model_class', model_class_name)

        model.fit(X_train, y_train)

        scores = evaluate(model, X_test, y_test)
        mlflow.log_metrics(scores)

        log_feature_importance(model)
        mlflow.sklearn.log_model(model, name='model')

        logger.info(f'Run {run_name} logged to MLflow')

    return scores['recall']


if __name__ == '__main__':
    main()
