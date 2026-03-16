import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import create_pipeline

df = pd.read_csv("../CustomerChurnDataset/03.model_processed.csv")

y = df["Churn"]

FEATURES = [
"Age",
"Gender",
"Tenure",
"Usage Frequency",
"Support Calls",
"Payment Delay",
"Subscription Type",
"Contract Length",
"Total Spend",
"Last Interaction",
"MonthlyCharges"
]
X = df[FEATURES]

pipeline = create_pipeline()

X_transformed = pipeline.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

pickle.dump(model, open("../artifacts/model.pkl", "wb"))
pickle.dump(pipeline, open("../artifacts/pipeline.pkl", "wb"))