import joblib

model = joblib.load("model_rf.pkl")   # yaha apna original model file name likhna
joblib.dump(model, "model_rf_compressed.pkl", compress=3)

print("Compressed model saved as model_rf_compressed.pkl")
