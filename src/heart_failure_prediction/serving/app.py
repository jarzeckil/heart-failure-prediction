from contextlib import asynccontextmanager
import logging
import os.path

from fastapi import FastAPI, HTTPException
import joblib
import numpy
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from heart_failure_prediction.config import MODEL_DIR
from heart_failure_prediction.serving.schemas import HeartDiseaseRecord

logger = logging.getLogger(__name__)

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_folder = MODEL_DIR
    model_name = 'model.joblib'
    model_path = os.path.join(model_folder, model_name)

    artifacts_path = os.path.join(MODEL_DIR, 'explainer_artifact')
    explainer_path = os.path.join(artifacts_path, 'explainer.joblib')
    features_path = os.path.join(artifacts_path, 'feature_names.joblib')

    try:
        model = joblib.load(model_path)
        artifacts['model'] = model
        logger.info('Model loaded successfully')
    except FileNotFoundError:
        logger.error(f"Couldn't read model from path {model_path}")

    try:
        explainer = joblib.load(explainer_path)
        artifacts['explainer'] = explainer
        logger.info('Explainer loaded successfully')
    except FileNotFoundError:
        logger.error(f"Couldn't read explainer from path {explainer_path}")

    try:
        feature_names = joblib.load(features_path)
        artifacts['feature_names'] = feature_names
        logger.info('Feature names loaded successfully')
    except FileNotFoundError:
        logger.error(f"Couldn't read feature names from path {features_path}")

    yield

    artifacts.clear()


app = FastAPI(lifespan=lifespan)


@app.get('/health')
async def health():
    model = artifacts.get('model')

    if model is None:
        raise HTTPException(status_code=503, detail='Service unavailable')

    return {'status': 'working'}


@app.post('/predict')
async def predict(record: HeartDiseaseRecord):
    model: Pipeline = artifacts.get('model')

    if model is None:
        logger.error("Model wasn't loaded")
        raise HTTPException(status_code=503, detail='Service unavailable')

    try:
        data = pd.DataFrame.from_records([record.model_dump()])
        pred = model.predict(data)
        pred_proba = model.predict_proba(data)

        return {
            'HeartDisease': int(pred[0]),
            'Probability-positive': float(pred_proba[0][1]),
            'Probability-negative': float(pred_proba[0][0]),
        }

    except Exception as e:
        logger.error(f'Error during prediction phase: {e}')
        raise HTTPException(status_code=500) from e


@app.post('/explain')
def explain(record: HeartDiseaseRecord):
    model: Pipeline = artifacts.get('model')
    explainer: shap.TreeExplainer = artifacts.get('explainer')
    feature_names: numpy.ndarray = artifacts.get('feature_names')

    if explainer is None or feature_names is None:
        logger.error("Artifacts weren't loaded")
        raise HTTPException(status_code=503, detail='Service unavailable')

    try:
        data = pd.DataFrame.from_records([record.model_dump()])
        preprocessor: ColumnTransformer = model.named_steps['preprocessing']
        X = preprocessor.transform(data)

        shap_values = explainer.shap_values(X)

        explanation = dict(zip(feature_names, shap_values.tolist()[0], strict=False))
        sorted_explanation = dict(
            sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        return sorted_explanation

    except Exception as e:
        logger.error(f'Error during prediction phase: {e}')
        raise HTTPException(status_code=500) from e
