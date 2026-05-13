from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Chargement du modèle
model = pickle.load(open('modele_spam_final.pkl', 'rb'))
tfidf = pickle.load(open('vectoriseur.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message.strip():
        return jsonify({'error': 'Message vide'}), 400

    vec = tfidf.transform([message]).toarray()
    prediction = int(model.predict(vec)[0])

    try:
        proba = model.predict_proba(vec)[0]
        spam_proba = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except:
        spam_proba = 1.0 if prediction == 1 else 0.0

    return jsonify({
        'prediction': prediction,
        'is_spam': prediction == 1,
        'spam_probability': round(spam_proba * 100, 1),
        'safe_probability': round((1 - spam_proba) * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
