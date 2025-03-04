import joblib
import numpy as np

model = joblib.load('models/logreg_model.joblib')
scaler = joblib.load('models/scaler.joblib')

score1 = 0.91
score2 = 1.02

input_features = np.array([[score1, score2]])
input_scaled = scaler.transform(input_features)
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

print(f"Prediction for score1={score1}, score2={score2}:")
print(f"Class: {prediction}")
print("Class probabilities:")
for i, cls in enumerate(model.classes_):
    print(f"  {cls}: {probabilities[i]:.4f}")