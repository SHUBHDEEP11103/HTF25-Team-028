import joblib
import numpy as np

def predict_new_asteroid(features_dict):
    """
    Predict if a new asteroid is hazardous using saved model.
    """
    # Load saved model, scaler, and feature names
    model = joblib.load('best_asteroid_classifier.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')

    # Create feature array (fill missing features with 0)
    features = np.array([[features_dict.get(name, 0) for name in feature_names]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    result = "🔴 HAZARDOUS" if prediction == 1 else "🟢 NON-HAZARDOUS"
    confidence = probability[prediction] * 100

    print(f"\n{'='*60}")
    print(f"🔮 PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nProbability Breakdown:")
    print(f"  Non-Hazardous: {probability[0]*100:.2f}%")
    print(f"  Hazardous:     {probability[1]*100:.2f}%")
    print(f"{'='*60}")

    return result, confidence, probability


if __name__ == "__main__":
    print("="*80)
    print("🚀 HAZARDOUS ASTEROID PREDICTION SYSTEM")
    print("="*80)

    # Load feature names so user knows what inputs are needed
    feature_names = joblib.load('feature_names.pkl')

    print("\n🪐 Please enter asteroid details:")
    print("(Press Enter to skip any feature — it will default to 0)\n")

    features_dict = {}
    for feature in feature_names[:10]:  # limit to first 10 important ones for simplicity
        try:
            val = input(f"{feature}: ")
            if val.strip() == "":
                features_dict[feature] = 0
            else:
                features_dict[feature] = float(val)
        except ValueError:
            print("Invalid input, setting to 0")
            features_dict[feature] = 0

    print("\n🔍 Running prediction...\n")
    predict_new_asteroid(features_dict)
