from flask import Flask, request, jsonify
import numpy as np
import joblib

# Charger les objets du modèle et du scaler
model = joblib.load('./artefacts/model.pkl')
scaler = joblib.load('./artefacts/scaler.pkl')
lambda_boxcox = joblib.load('./artefacts/lambda_boxcox.pkl')

app = Flask(__name__)

# Route de santé
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête JSON
        data = request.get_json()

        # Vérification de la présence de toutes les clés dans les données
        required_keys = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return jsonify({"error": f"Missing data for the following fields: {', '.join(missing_keys)}"}), 400
        
        # Extraire les variables du JSON et les convertir en float
        ZN = float(data['ZN'])
        INDUS = float(data['INDUS'])
        CHAS = float(data['CHAS'])
        NOX = float(data['NOX'])
        RM = float(data['RM'])
        AGE = float(data['AGE'])
        DIS = float(data['DIS'])
        TAX = float(data['TAX'])
        PTRATIO = float(data['PTRATIO'])
        B = float(data['B'])
        LSTAT = float(data['LSTAT'])
        
    except (ValueError, TypeError) as e:
        return jsonify({"error": "Please provide valid numeric inputs for all fields."}), 400

    # Convertir les variables en tableau numpy
    features = np.array([ZN, INDUS, CHAS, NOX, RM, AGE, DIS, TAX, PTRATIO, B, LSTAT]).reshape(1, -1)

    # Appliquer le scaler (standardisation) aux features
    features_scaled = scaler.transform(features)

    # Prédiction sur la variable transformée
    y_pred_transformed = model.predict(features_scaled)

    # Inverser la transformation Box-Cox
    if lambda_boxcox != 0:
        y_pred_real = (y_pred_transformed * lambda_boxcox + 1) ** (1 / lambda_boxcox)
    else:
        y_pred_real = np.exp(y_pred_transformed)

    # Limiter le résultat à 3 décimales
    y_pred_real_rounded = round(y_pred_real[0], 3)

    # Renvoyer la prédiction
    return jsonify({"prediction": y_pred_real_rounded})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
