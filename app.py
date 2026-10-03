from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Devoir libre • Stocks ABC", page_icon="📦", layout="wide")
st.markdown("<style>" + (Path(__file__).parent / "style.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="eyebrow">Devoir libre · Analyse décisionnelle</div>
  <h1>Chaque article, à sa juste priorité.</h1>
  <p>Explorez votre inventaire, construisez le classement ABC et comparez les modèles.</p>
  <span class="pill">8 attributs → 5 critères → 3 classes</span>
</div>
""", unsafe_allow_html=True)

CONVERSIONS = {
    "Risk": {"High": .47, "Normal": .35, "Low": .18},
    "Demand fluctuation": {"Increasing": .36, "Stable": .28, "Unknown": .20, "Decreasing": .16, "Ending": 0},
    "Consignment stock": {"No": .80, "Yes": .20},
    "Unit size": {"Large": .53, "Medium": .31, "Small": .13},
}
NUMERIQUES = ["Average stock", "Daily usage", "Unit cost", "Lead time"]
COLONNES = list(CONVERSIONS) + NUMERIQUES
TITRES = [
    "1 • Identifier les variables", "2 • Convertir les variables qualitatives",
    "3 • Construire les critères", "4 • Calculer TOPSIS", "5 • Classer les articles",
    "6 • Appliquer ABC", "7 • Exporter la base", "8 • Entraîner les modèles",
    "9 • Comparer les performances",
]

def normaliser(serie):
    amplitude = serie.max() - serie.min()
    return (serie - serie.min()) / amplitude if amplitude else serie * 0

def attribuer_classes(score):
    rang = score.rank(ascending=False, method="first").astype(int)
    n = len(score)
    classes = np.where(rang <= int(.2 * n), "A",
                       np.where(rang <= int(.5 * n), "B", "C"))
    return rang, classes

def poids_entropie(criteres):
    # Proportions par colonne ; une colonne nulle est uniforme.
    sommes = criteres.sum().replace(0, np.nan)
    p = criteres.div(sommes).fillna(1 / len(criteres))
    # Convention : 0 * log(0) = 0.
    termes = p * np.log(p.where(p > 0, 1))
    entropie = (-termes.sum() / np.log(len(criteres))).clip(0, 1)
    diversite = (1 - entropie).where(criteres.nunique() > 1, 0)
    if diversite.sum() <= 1e-12:
        return np.full(criteres.shape[1], 1 / criteres.shape[1])
    return (diversite / diversite.sum()).to_numpy()

@st.cache_data
def calculer(df, methode):
    scores = df.copy()
    for col, correspondance in CONVERSIONS.items():
        scores[col] = df[col].map(correspondance)
    normes = scores.copy()
    for col in NUMERIQUES:
        normes[col] = normaliser(scores[col])
    criteres = pd.DataFrame({
        "Criticality": .78 * scores["Risk"] + .22 * scores["Demand fluctuation"],
        "Demand": .71 * normes["Daily usage"] + .29 * normes["Average stock"],
        "Supply": .75 * normes["Lead time"] + .25 * scores["Consignment stock"],
        "Unit cost": normes["Unit cost"],
        "Unit size": scores["Unit size"],
    })
    if methode == "Entropie":
        poids = poids_entropie(criteres)
    elif methode == "Kartal (2016)":
        poids = np.array([.33, .15, .18, .12, .22])
    else:
        poids = np.full(5, .2)
    diviseur = np.sqrt((criteres ** 2).sum()).replace(0, 1)
    r = criteres / diviseur
    v = r * poids
    ideal, anti = v.max(), v.min()
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((v - anti) ** 2).sum(axis=1))
    total = dp + dm
    score = (dm / total.replace(0, np.nan)).fillna(.5)
    rang, classes = attribuer_classes(score)
    resultat = df.copy()
    resultat.insert(0, "Article", np.arange(1, len(df) + 1))
    resultat["Score_TOPSIS"] = score
    resultat["Rang"] = rang
    resultat["Classe"] = classes
    return scores, criteres, poids, r, v, resultat

@st.cache_data
def entrainer(scores, classes):
    X_train, X_test, y_train, y_test = train_test_split(
        scores, classes, test_size=.30, random_state=42, stratify=classes
    )
    modeles = {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    lignes, matrices = [], {}
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        prediction = modele.predict(X_test)
        precision, rappel, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="macro", zero_division=0
        )
        lignes.append({"Modèle": nom, "Accuracy": accuracy_score(y_test, prediction),
                       "Précision macro": precision, "Rappel macro": rappel, "F1 macro": f1})
        matrices[nom] = pd.DataFrame(
            confusion_matrix(y_test, prediction, labels=["A", "B", "C"]),
            index=["A", "B", "C"], columns=["A", "B", "C"]
        )
    return pd.DataFrame(lignes).sort_values("F1 macro", ascending=False), matrices, len(X_train), len(X_test)

def telecharger(tableau, nom):
    st.download_button("Télécharger le CSV", tableau.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nom, mime="text/csv", key=nom)

def code_simple(code):
    with st.expander("Voir le code simple"):
        st.code(code, language="python")

st.sidebar.markdown('<div class="brand">Stock<span>Lab</span></div><div class="sidebar-sub">Gestion des stocks multi-attributs</div>', unsafe_allow_html=True)
st.sidebar.caption("PARCOURS DU DEVOIR")
question = st.sidebar.radio("Choisir une question", TITRES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("DONNÉES & PONDÉRATION")
methode = st.sidebar.selectbox("Pondération des cinq critères", ["Entropie", "Kartal (2016)", "Poids égaux"])
st.sidebar.caption("Entropie : poids calculés à partir des données. Les poids internes de la Table 2 restent inchangés.")
try:
    df = pd.read_csv(Path(__file__).parent / "inventory_data.csv")
except Exception as erreur:
    st.error(f"Impossible de lire le CSV : {erreur}")
    st.stop()
manquantes = set(COLONNES) - set(df.columns)
if manquantes:
    st.error("Colonnes manquantes : " + ", ".join(sorted(manquantes)))
    st.stop()
df = df[COLONNES].copy()
if len(df) < 10:
    st.error("Utilisez au moins 10 articles pour cette analyse.")
    st.stop()
for col in NUMERIQUES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df.isna().any().any() or not np.isfinite(df[NUMERIQUES].to_numpy()).all():
    st.error("Le fichier contient des données manquantes ou des valeurs numériques invalides.")
    st.stop()
for col, correspondance in CONVERSIONS.items():
    inconnues = set(df[col]) - set(correspondance)
    if inconnues:
        st.error(f"Catégories inconnues dans {col} : {', '.join(map(str, inconnues))}")
        st.stop()
scores, criteres, poids, r, v, resultat = calculer(df, methode)
classement = resultat.sort_values("Rang")
a, b, c = st.columns(3)
a.metric("Articles", len(df))
b.metric("Variables", len(df.columns))
c.metric("Valeurs manquantes", int(df.isna().sum().sum()))
q = TITRES.index(question) + 1
st.progress(q / len(TITRES), text=f"Question {q} sur {len(TITRES)} · {methode}")
st.markdown(f'<div class="step"><div class="step-number">{q:02d}</div><div><small>COMPRENDRE ET APPLIQUER</small><h2>{question.split(" • ", 1)[1]}</h2></div></div>', unsafe_allow_html=True)

if q == 1:
    st.write("Les variables qualitatives décrivent des catégories. Les quantitatives représentent des mesures.")
    st.dataframe(pd.DataFrame({"Variable": df.columns,
                              "Nature": ["Qualitative" if col in CONVERSIONS else "Quantitative" for col in df]}),
                 hide_index=True, width="stretch")
    st.dataframe(df.head(10), width="stretch")
    code_simple('quali = df.select_dtypes(include="object").columns\nquanti = df.select_dtypes(include="number").columns\nprint(quali)\nprint(quanti)')
elif q == 2:
    st.write("On remplace chaque catégorie par le score normalisé de la Table 1 du devoir.")
    lignes = [{"Variable": col, "Catégorie": cat, "Score": score}
              for col, correspondance in CONVERSIONS.items() for cat, score in correspondance.items()]
    st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
    st.caption("Les scores de taille sont repris exactement, y compris Medium = 0.31.")
    st.dataframe(scores.head(10), width="stretch")
    code_simple('df_scores = df.copy()\ndf_scores["Risk"] = df["Risk"].map({"High": 0.47, "Normal": 0.35, "Low": 0.18})')
    telecharger(scores, "scores_qualitatifs_app.csv")
elif q == 3:
    st.write("On normalise les quantités par (x − minimum) / (maximum − minimum), puis on combine les attributs avec les poids de la Table 2.")
    st.code("Criticality = 0.78 × Risk + 0.22 × Demand fluctuation\nDemand = 0.71 × Daily usage normalisée + 0.29 × Average stock normalisé\nSupply = 0.75 × Lead time normalisé + 0.25 × Consignment stock", language=None)
    st.info("Comme dans Kartal, on conserve aussi Unit cost et Unit size : cinq critères finaux utilisent les huit attributs.")
    st.dataframe(criteres.head(10), width="stretch")
    code_simple('criteres["Criticality"] = 0.78 * scores["Risk"] + 0.22 * scores["Demand fluctuation"]')
elif q == 4:
    if methode == "Entropie":
        st.write("p = x / somme(x). Entropie : e = -somme(p × ln(p)) / ln(n). Poids : w = (1 - e) / somme(1 - e).")
        st.info("Un critère plus discriminant reçoit plus de poids. Cela mesure la dispersion, pas l’importance métier. Un critère constant reçoit un poids nul ; si tous sont constants, on utilise des poids égaux.")
        code_simple('p = criteres / criteres.sum()\nentropie = -(p * np.log(p.where(p > 0, 1))).sum() / np.log(len(criteres))\ndiversite = 1 - entropie\npoids = diversite / diversite.sum()')
    st.write("TOPSIS mesure la proximité avec un article idéal. Ici, une valeur élevée de chaque critère augmente la priorité de suivi.")
    st.dataframe(pd.DataFrame({"Critère": criteres.columns, "Poids": poids}),
                 hide_index=True, width="stretch")
    st.caption("Les poids finaux sont distincts des poids internes de la question 3.")
    gauche, droite = st.columns(2)
    gauche.write("Matrice normalisée")
    gauche.dataframe(r.head(), width="stretch")
    droite.write("Matrice pondérée")
    droite.dataframe(v.head(), width="stretch")
    st.latex(r"C_i = \frac{D_i^-}{D_i^+ + D_i^-}")
    st.write("D+ : distance à l'idéal positif (maximums). D− : distance à l'idéal négatif (minimums). Score entre 0 et 1.")
    st.dataframe(resultat[["Article", "Score_TOPSIS"]].head(10), hide_index=True)
    code_simple('r = criteres / np.sqrt((criteres ** 2).sum())\nv = r * poids\nd_plus = np.sqrt(((v - v.max()) ** 2).sum(axis=1))\nd_moins = np.sqrt(((v - v.min()) ** 2).sum(axis=1))\nscore = d_moins / (d_plus + d_moins)')
elif q == 5:
    st.write("Les articles sont triés du plus grand au plus petit score. Le rang 1 correspond à la priorité la plus élevée.")
    st.dataframe(classement, hide_index=True, width="stretch", column_config={
        "Score_TOPSIS": st.column_config.ProgressColumn("Score TOPSIS", min_value=0, max_value=1, format="%.3f")
    })
    code_simple('classement = resultat.sort_values("Score_TOPSIS", ascending=False)')
    telecharger(classement, "classement_topsis_app.csv")
elif q in (6, 7):
    st.write("A : premiers 20 % des articles. B : suivants 30 %. C : derniers 50 %. Les seuils sont arrondis vers le bas si nécessaire.")
    effectifs = resultat["Classe"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    cartes = st.columns(3)
    couleurs = ["#0f766e", "#d97706", "#64748b"]
    descriptions = ["Priorité élevée · suivi rigoureux", "Priorité intermédiaire · suivi modéré", "Priorité faible · gestion simplifiée"]
    for carte, classe, couleur, description in zip(cartes, ["A", "B", "C"], couleurs, descriptions):
        carte.markdown(f'<div class="abc-card" style="--accent:{couleur}"><b>CLASSE {classe}</b><strong>{effectifs[classe]}</strong><p>{description}</p></div>', unsafe_allow_html=True)
    graphique = effectifs.rename_axis("Classe").reset_index(name="Articles")
    st.altair_chart(alt.Chart(graphique).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Classe:N", axis=alt.Axis(labelAngle=0)),
        y="Articles:Q",
        color=alt.Color("Classe:N", scale=alt.Scale(domain=["A", "B", "C"], range=couleurs), legend=None),
        tooltip=["Classe", "Articles"]
    ).properties(height=230), width="stretch")
    st.dataframe(effectifs.rename("Nombre").to_frame(), width="stretch")
    st.info("Ces proportions sont celles du nombre d'articles ; elles ne garantissent pas 80 %, 15 % et 5 % de la valeur financière. Kartal utilise aussi cette répartition.")
    if q == 7:
        st.write("La base ci-dessous conserve l'ordre initial et ajoute Article, Score_TOPSIS, Rang et Classe.")
        st.dataframe(resultat, hide_index=True, width="stretch")
        telecharger(resultat, "inventory_avec_classes_app.csv")
    code_simple('classement["Classe"] = "C"\nclassement.loc[classement["Rang"] <= int(0.5 * len(df)), "Classe"] = "B"\nclassement.loc[classement["Rang"] <= int(0.2 * len(df)), "Classe"] = "A"')
elif q in (8, 9):
    st.write("Les huit attributs servent à prédire Classe. Le score TOPSIS, le rang et l'identifiant sont exclus des entrées.")
    st.caption("Les labels sont construits sur cet inventaire complet : le test mesure leur reproduction, pas la validation de priorités par un expert.")
    if resultat["Classe"].value_counts().min() < 2:
        st.error("Chaque classe doit contenir au moins deux articles pour la division stratifiée.")
        st.stop()
    with st.spinner("Entraînement des trois modèles…"):
        comparaison, matrices, n_train, n_test = entrainer(scores, resultat["Classe"])
    st.write(f"Division stratifiée : {n_train} articles d'entraînement et {n_test} de test. Graine aléatoire : 42.")
    if q == 8:
        st.write("Régression logistique, arbre de décision et forêt aléatoire. La standardisation de la régression est apprise uniquement sur l'entraînement.")
        st.success("Les trois modèles ont été entraînés. Ouvrez la question 9 pour comparer les résultats.")
        code_simple('X_train, X_test, y_train, y_test = train_test_split(\n    scores, resultat["Classe"], test_size=0.30, random_state=42, stratify=resultat["Classe"]\n)\nmodele.fit(X_train, y_train)\nprediction = modele.predict(X_test)')
    else:
        st.write("Accuracy : proportion correcte. Précision : fiabilité des prédictions. Rappel : articles retrouvés. F1 : équilibre précision/rappel. Macro donne le même poids à chaque classe.")
        st.dataframe(comparaison, hide_index=True, width="stretch", column_config={
            col: st.column_config.ProgressColumn(col, min_value=0, max_value=1, format="%.3f")
            for col in ["Accuracy", "Précision macro", "Rappel macro", "F1 macro"]
        })
        st.altair_chart(alt.Chart(comparaison).mark_bar(cornerRadiusEnd=5, color="#0f766e").encode(
            x=alt.X("F1 macro:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Modèle:N", sort="-x"), tooltip=["Modèle", "Accuracy", "F1 macro"]
        ).properties(height=170), width="stretch")
        st.success("Meilleur F1 sur ce test : " + comparaison.iloc[0]["Modèle"])
        nom = st.selectbox("Matrice de confusion du modèle", list(matrices))
        st.caption("Lignes : classes réelles. Colonnes : classes prédites. La diagonale contient les prédictions correctes.")
        st.dataframe(matrices[nom], width="stretch")
        telecharger(comparaison, "comparaison_modeles_app.csv")
st.divider()
st.caption("Référence : Kartal et al. (2016), DOI 10.1016/j.cie.2016.06.004. L'article applique SAW, AHP et VIKOR ; cette application suit le devoir en appliquant TOPSIS.")



