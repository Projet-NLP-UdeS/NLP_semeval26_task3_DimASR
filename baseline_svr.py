import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR

def run_svr_baseline(train_df, dev_df, max_features=5000):
    """
    Entraîne un modèle SVR avec TF-IDF et retourne les prédictions.
    """
    print("Préparation des données pour le SVR...")
    
    # Préparation des inputs : concaténation du texte et de l'aspect
    # fillna('') si valeurs nulles
    train_texts = train_df['Text'].fillna('') + " " + train_df['Aspect'].fillna('')
    dev_texts = dev_df['Text'].fillna('') + " " + dev_df['Aspect'].fillna('')

    # Extraction des labels (Gold standards)
    gold_v_train = train_df['Valence'].values.astype(float)
    gold_a_train = train_df['Arousal'].values.astype(float)
    
    gold_v_dev = dev_df['Valence'].values.astype(float)
    gold_a_dev = dev_df['Arousal'].values.astype(float)

    # Vectorisation TF-IDF
    print(f"Vectorisation TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)

    # Entraînement des modèles
    print("Entraînement des modèles SVR (Valence et Arousal)...")
    model_v = SVR()
    model_a = SVR()
    
    model_v.fit(X_train, gold_v_train)
    model_a.fit(X_train, gold_a_train)

    # Prédictions sur le set de développement
    print("Génération des prédictions SVR...")
    pred_v = model_v.predict(X_dev)
    pred_a = model_a.predict(X_dev)

    # On retourne les prédictions et les vrais labels pour l'évaluation globale
    return pred_v, pred_a, gold_v_dev, gold_a_dev