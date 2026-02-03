import numpy as np
import joblib
import os

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model_heart.joblib"))
    return model

def predict_fn(input_data, model):
    x = np.array(input_data)

    # NORMALIZE
    x_norm = (x - model['mu']) / model['sigma']

    z = np.dot(x_norm, model['w']) + model['b']
    prob = 1 / (1 + np.exp(-z))

    return prob.tolist()

print("Inference function created successfully")
