from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_churn

app = FastAPI()

class Customer(BaseModel):
    Age:int
    Gender:int
    Tenure:int
    UsageFrequency:int
    SupportCalls:int
    PaymentDelay:int
    SubscriptionType:int
    ContractLength:int
    TotalSpend:int
    LastInteraction:int
    MonthlyCharges:float


@app.get("/")
def home():
    return {"message":"Customer Churn Prediction API"}

@app.post("/predict")
def predict(customer:Customer):
    features = [
        customer.Age,
        customer.Gender,
        customer.Tenure,
        customer.UsageFrequency,
        customer.SupportCalls,
        customer.PaymentDelay,
        customer.SubscriptionType,
        customer.ContractLength,
        customer.TotalSpend,
        customer.LastInteraction,
        customer.MonthlyCharges
    ]

    prediction = predict_churn(features)

    return {"churn_prediction":prediction}