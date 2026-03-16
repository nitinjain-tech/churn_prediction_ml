import pickle
import numpy as np

model = pickle.load(open("artifacts/model.pkl","rb"))
pipeline = pickle.load(open("artifacts/pipeline.pkl","rb"))

def predict_churn(data):

    data = np.array(data).reshape(1,-1)

    transformed = pipeline.transform(data)

    prediction = model.predict(transformed)

    return int(prediction[0])