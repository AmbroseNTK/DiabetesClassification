from tensorflow.keras.models import load_model
from joblib import dump, load
import pandas as pd

class ClassificationModel:
    def __init__(self, model_name, model, custom_fit_fn=None, custom_evaluate_fn=None, skip=False):
        self.model_name = model_name
        self.model = model
        self.custom_fit_fn = custom_fit_fn
        self.custom_evaluate_fn = custom_evaluate_fn
        self.skip = skip
        self.describes = []

    @staticmethod
    def from_file(file_path, model_name,describe_path, keras_model=False):
        model_name = model_name
        model = None
        if keras_model:
            model = load_model(file_path)
        else:
            model = load(file_path)
        cm = ClassificationModel(model_name, model)
        cm.describes = pd.read_csv(describe_path)
        return cm
