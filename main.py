import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(page_title="DiamondScope", layout="wide", page_icon="💎")

THEME_CSS = """
<style>
body, .main, .css-18e3th9, .css-1outpf7 {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 40%, #f1f5f9 100%) !important;
    color: #0f172a !important;
    background-attachment: fixed !important;
}
.stButton>button {
    background-color: #2563eb !important;
    color: blue !important;
    border: none !important;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    color: #0f172a !important;
}
.css-1lsmgbg.egzxvld1 {
    background-color: rgba(255,255,255,0.95) !important;
}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #2563eb !important;
}
.stMetricValue, .stMetricDelta {
    color: #0f172a !important;
}
</style>
"""

def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)

COLOR_CATEGORIES = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_CATEGORIES = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
MODEL_COLOR_CATEGORIES = ["E", "F", "G", "H", "I", "J"]
MODEL_CLARITY_CATEGORIES = ["VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
CUT_CATEGORIES = ["Ideal", "Premium", "Very Good", "Good", "Fair"]


@st.cache_resource(show_spinner=False)
def load_models():
    classif = joblib.load("model_scaler.pkl")
    regression = joblib.load("modellr_scalerlr.pkl")
    model_knn = classif["model"]
    scaler_knn = classif["scaler"]
    model_lr = regression["model_lr"]
    scaler_lr = regression["scaler_lr"]
    return model_knn, scaler_knn, model_lr, scaler_lr


@st.cache_data(show_spinner=False)
def load_data():
    df = sns.load_dataset("diamonds")
    df = df.drop(columns=["x", "y", "z"], errors="ignore")
    df["coupe_premium"] = (df["cut"] == "Premium").astype(int)
    return df


def build_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["carat"] = df["carat"]
    data["depth"] = df["depth"]
    data["table"] = df["table"]

    color_dummies = pd.get_dummies(df["color"], prefix="color")
    clarity_dummies = pd.get_dummies(df["clarity"], prefix="clarity")
    data = pd.concat([data, color_dummies, clarity_dummies], axis=1)

    for color in MODEL_COLOR_CATEGORIES:
        if f"color_{color}" not in data.columns:
            data[f"color_{color}"] = 0
    for clarity in MODEL_CLARITY_CATEGORIES:
        if f"clarity_{clarity}" not in data.columns:
            data[f"clarity_{clarity}"] = 0

    feature_order = ["carat", "depth", "table"] + [f"color_{c}" for c in MODEL_COLOR_CATEGORIES] + [f"clarity_{c}" for c in MODEL_CLARITY_CATEGORIES]
    return data.reindex(columns=feature_order, fill_value=0)


def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data["carat"] = df["carat"]
    data["depth"] = df["depth"]
    data["table"] = df["table"]

    cut_dummies = pd.get_dummies(df["cut"], prefix="cut")
    color_dummies = pd.get_dummies(df["color"], prefix="color")
    clarity_dummies = pd.get_dummies(df["clarity"], prefix="clarity")
    data = pd.concat([data, cut_dummies, color_dummies, clarity_dummies], axis=1)

    cut_keys = ["cut_Premium", "cut_Very Good", "cut_Good", "cut_Fair"]
    color_keys = [f"color_{c}" for c in MODEL_COLOR_CATEGORIES]
    clarity_keys = [f"clarity_{c}" for c in MODEL_CLARITY_CATEGORIES]

    for key in cut_keys + color_keys + clarity_keys:
        if key not in data.columns:
            data[key] = 0

    feature_order = ["carat", "depth", "table"] + cut_keys + color_keys + clarity_keys
    return data.reindex(columns=feature_order, fill_value=0)


def get_inputs():
    st.sidebar.header("Paramètres du diamant")
    carat = st.sidebar.number_input("Carat", min_value=0.20, max_value=5.00, value=1.00, step=0.01, format="%.2f")
    depth = st.sidebar.number_input("Depth", min_value=50.0, max_value=80.0, value=61.5, step=0.1, format="%.1f")
    table = st.sidebar.number_input("Table", min_value=50.0, max_value=80.0, value=55.0, step=0.1, format="%.1f")
    color = st.sidebar.selectbox("Color", COLOR_CATEGORIES, index=2)
    clarity = st.sidebar.selectbox("Clarity", CLARITY_CATEGORIES, index=3)
    cut = st.sidebar.selectbox("Cut", CUT_CATEGORIES, index=0)
    return carat, depth, table, color, clarity, cut


def run_price_page(model_lr, scaler_lr, data):
    st.header("Prédiction du prix du diamant")
    st.write("Utilisez les entrées de la barre latérale pour estimer le prix en temps réel.")
    sample = pd.DataFrame([{
        "carat": data[0],
        "depth": data[1],
        "table": data[2],
        "cut": data[5],
        "color": data[3],
        "clarity": data[4],
    }])

    features = build_regression_features(sample)
    prediction = model_lr.predict(features)[0]
    st.metric("Prix estimé", f"{prediction:,.0f} USD")

    with st.expander("Voir les caractéristiques encodées utilisées par le modèle"):
        st.dataframe(features.T)

    st.markdown("---")
    st.subheader("Hypothèses de modélisation")
    st.write(
        "Le modèle de régression utilise le carat, la profondeur, la table, la couleur, la clarté et des indicateurs de coupe. "
        "La coupe `Ideal` est la catégorie de référence pour le modèle."
    )


def run_premium_page(model_knn, scaler_knn, data):
    st.header("Prédiction de la catégorie `coupe_premium`")
    st.write("Prédisez si un diamant appartient à la catégorie `Premium` selon ses caractéristiques.")
    sample = pd.DataFrame([{
        "carat": data[0],
        "depth": data[1],
        "table": data[2],
        "color": data[3],
        "clarity": data[4],
    }])
    features = build_classification_features(sample)
    X_scaled = scaler_knn.transform(features)

    prediction = model_knn.predict(X_scaled)[0]
    proba = model_knn.predict_proba(X_scaled)[0][1]
    label = "Oui" if prediction == 1 else "Non"

    st.metric("Coupe Premium prédite", label)
    st.metric("Probabilité Premium", f"{proba * 100:.1f}%")

    with st.expander("Voir les caractéristiques encodées utilisées par le modèle"):
        st.dataframe(features.T)

    st.markdown("---")
    st.subheader("Remarque")
    st.write(
        "Le modèle KNN prédit `coupe_premium` sur la base du carat, de la profondeur, de la table, de la couleur et de la clarté. "
        "La probabilité représente la confiance du modèle pour la classe `Premium`."
    )


def run_analysis_page(model_knn, scaler_knn, diamonds):
    st.header("Analyse des données et performances")
    st.write("Exploration du jeu de données `diamonds` et évaluation du modèle de classification.")

    st.subheader("Vue d’ensemble des données")
    st.write(f"Jeu de données chargé : {diamonds.shape[0]:,} diamants.")
    st.write(diamonds[["carat", "cut", "color", "clarity", "depth", "table", "price"]].head(7))

    st.subheader("Distribution de la qualité de taille")
    st.bar_chart(diamonds["cut"].value_counts())

    st.subheader("Corrélation des variables numériques")
    corr = diamonds[["price", "carat", "depth", "table"]].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Performance du modèle `coupe_premium`")
    X = build_classification_features(diamonds)
    X_scaled = scaler_knn.transform(X)
    y_true = diamonds["coupe_premium"].values
    y_pred = model_knn.predict(X_scaled)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Précision", f"{prec:.3f}")
    st.metric("Rappel", f"{rec:.3f}")
    st.metric("F1-score", f"{f1:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, xticklabels=["Non", "Oui"], yticklabels=["Non", "Oui"])
    ax2.set_xlabel("Prédiction")
    ax2.set_ylabel("Réel")
    ax2.set_title("Matrice de confusion pour `coupe_premium`")
    st.pyplot(fig2)

    st.markdown("---")
    st.write(
        "La matrice de confusion montre combien de diamants ont été correctement identifiés comme `Premium` ou non. "
        "L’analyse de ces erreurs aide à comprendre les limites du modèle KNN."
    )


def run_app():
    st.title("💎 DiamondScope")
    st.write(
        "Bienvenue sur DiamondScope : un prototype Streamlit pour estimer le prix d’un diamant, "
        "prédire si sa coupe est `Premium`, et analyser les données d’entraînement."
    )
    st.markdown("---")

    page = st.sidebar.radio("Navigation", ["Accueil", "Prix diamant", "Coupe Premium", "Analyse"])
    st.sidebar.markdown("### ⚙️ Options rapides")
    st.sidebar.write("Utilisez les contrôles sur la gauche pour mettre à jour les prédictions en direct.")
    model_knn, scaler_knn, model_lr, scaler_lr = load_models()
    diamonds = load_data()
    carat, depth, table, color, clarity, cut = get_inputs()

    if page == "Accueil":
        st.header("Présentation")
        st.write(
            "Sélectionnez une page dans la barre latérale pour :\n"
            "- estimer un prix,\n"
            "- prédire si le diamant est `Premium`,\n"
            "- explorer les performances et visualisations."
        )

        with st.expander("Pourquoi ce prototype ?"):
            st.write(
                "Ce dashboard est conçu pour rendre les prédictions actionnables en joaillerie, "
                "tout en mettant en valeur la qualité du modèle et les relations entre carat, couleur, clarté et prix."
            )

        st.markdown("---")
        st.write("### 📌 Astuce")
        st.write("Modifiez les paramètres dans la barre latérale pour voir les prédictions se recalculer instantanément.")
        st.success("L’interface utilise un style sombre, des emojis et des encadrés pour une expérience plus dynamique.")

    elif page == "Prix diamant":
        run_price_page(model_lr, scaler_lr, (carat, depth, table, color, clarity, cut))
    elif page == "Coupe Premium":
        run_premium_page(model_knn, scaler_knn, (carat, depth, table, color, clarity, cut))
    elif page == "Analyse":
        run_analysis_page(model_knn, scaler_knn, diamonds)


if __name__ == "__main__":
    run_app()

