import pickle
import json
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    global model
    model_path = Model.get_model_path(model_name = 'AutoMLb9b2f9558best') # this name is model.id of model that we want to deploy
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

input_sample = pd.DataFrame(data=[{
              "make": "alfa-romero",         # This is a decimal type sample. Use the data type that reflects this column in your data
              "fuel-type": "gas",    # This is a string type sample. Use the data type that reflects this column in your data
              "aspiration": "std",
              "num-of-doors": "two",
              "body-style": "convertible",
              "drive-wheels": "rwd",
              "engine-location": "front",
              "wheel-base": 88.6,
              "length": 168.8,
              "width": 64.1,
              "height": 48.8,
              "curb-weight": 2548,
              "engine-type": "dohc",
              "num-of-cylinders": "four",
              "engine-size": 130,
              "fuel-system": "mpfi",
              "bore": 3.47,
              "stroke": 2.68,
              "compression-ratio": 9,
              "horsepower": 111,
              "peak-rpm": 5000,
              "city-mpg": 21,
              "highway-mpg": 27
            }])

output_sample = np.array([0])              # This is a integer type sample. Use the data type that reflects the expected result

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error