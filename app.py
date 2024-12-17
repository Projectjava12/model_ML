from flask import Flask, request, jsonify, render_template
import joblib

# Charger les modèles et leurs performances
models = joblib.load("model_prediction_churn.pkl")

# Sélectionner un modèle par défaut (par exemple, XGBC)
selected_model_key = "XGBC"  # Remplacez par un autre modèle si nécessaire
selected_model = models[selected_model_key]["model"]

app = Flask(__name__)

@app.route("/")
def index():
    # Initialisation sans résultat
    return render_template("index.html", prediction=None, probability=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
    
        # Récupérer les features envoyées depuis le formulaire
        features = [float(request.form[f"feature{i}"]) for i in range(1, 25)]
        
        # Effectuer la prédiction avec le modèle par défaut
        prediction = selected_model.predict([features])[0]
        probability = selected_model.predict_proba([features])[0][1] * 100  # Probabilité de churn (%)
        
        # Retourner la page avec les résultats
        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability, 2),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

