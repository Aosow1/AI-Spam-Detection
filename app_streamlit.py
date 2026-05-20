import streamlit as st
import pickle
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détection de Spam",
    page_icon="🛡️",
    layout="centered"
)

# Chargement du modèle
@st.cache_resource
def load_model():
    model = pickle.load(open('modele_spam_final.pkl', 'rb'))
    tfidf = pickle.load(open('vectoriseur.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()

# CSS personnalisé
st.markdown("""
<style>
    .main { max-width: 700px; margin: auto; }
    .spam-box {
        background: #fff0f0;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .safe-box {
        background: #f0fff4;
        border: 2px solid #21c55d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-title { font-size: 28px; font-weight: bold; margin: 0; }
    .subtitle { color: #666; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ Spam Detector")
st.markdown('<p class="subtitle">Entrez un message pour savoir s\'il est un spam ou non.</p>', unsafe_allow_html=True)

# Stats dans la sidebar
with st.sidebar:
    st.header("📊 À propos")
    st.info("Ce modèle a été entraîné pour détecter les messages spam avec une bonneprécision.")
    st.metric("Modèle", "TF-IDF + Classifier")
    if st.button("🔄 Réinitialiser"):
        st.rerun()

# Zone de saisie
message = st.text_area(
    "✉️ Message à analyser",
    height=150,
    placeholder="Ex: Félicitations ! Vous avez gagné 1000€. Cliquez ici..."
)

# Bouton d'analyse
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyser = st.button("🔍 Analyser le message", use_container_width=True, type="primary")

# Résultat
if analyser and message.strip():
    with st.spinner("Analyse en cours..."):
        vec = tfidf.transform([message]).toarray()
        prediction = model.predict(vec)[0]

        # Probabilités si disponibles
        try:
            proba = model.predict_proba(vec)[0]
            spam_proba = proba[1] if len(proba) > 1 else proba[0]
            safe_proba = 1 - spam_proba
        except:
            spam_proba = 1.0 if prediction == 1 else 0.0
            safe_proba = 1.0 - spam_proba

        st.divider()

        if prediction == 1:
            st.markdown(f"""
            <div class="spam-box">
                <p class="result-title">🚨 SPAM détecté !</p>
                <p>Ce message est probablement un spam.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <p class="result-title">✅ Message sain</p>
                <p>Ce message ne semble pas être un spam.</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Barres de probabilité
        st.subheader("📈 Probabilités")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🚨 Spam", f"{spam_proba*100:.1f}%")
            st.progress(float(spam_proba))
        with col_b:
            st.metric("✅ Pas spam", f"{safe_proba*100:.1f}%")
            st.progress(float(safe_proba))

elif analyser and not message.strip():
    st.warning("⚠️ Veuillez entrer un message avant d'analyser.")


st.divider()
st.subheader("💡 Exemples à tester")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Exemple spam", use_container_width=True):
        st.session_state['exemple'] = "URGENT: Vous avez gagné un iPhone! Cliquez maintenant: www.gagnant-telephonegn.com"
with col2:
    if st.button("✅ Exemple normal", use_container_width=True):
        st.session_state['exemple'] = "Salut, on se retrouve demain à 14h pour la réunion ?"

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#aaa; font-size:13px;">Détection de Spam — Projet IA</p>', unsafe_allow_html=True)
