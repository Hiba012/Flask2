from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('ranfor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Récupération des valeurs du formulaire
        temp = float(request.form['temperature'])
        hum = float(request.form['humidity'])
        wind = float(request.form['wind_speed'])
        cloud = float(request.form['cloud_cover'])
        press = float(request.form['pressure'])

        # Création du tableau de données
        data = np.array([[temp, hum, wind, cloud, press]])

        # Prédiction
        pred = model.predict(data)
        prediction = "🌧️ Il va pleuvoir." if pred[0] == 1 else "☀️ Pas de pluie."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
