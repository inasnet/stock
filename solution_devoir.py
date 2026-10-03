from pathlib import Path
### 1
import pandas as pd

# Charger le fichier
df = pd.read_csv(Path(__file__).parent / "inventory_data.csv")

# Afficher les premières lignes
print(df.head())

# Identifier les variables qualitatives
quali = df.select_dtypes(include="object").columns
print("Variables qualitatives :", quali.tolist())

# Identifier les variables quantitatives
quanti = df.select_dtypes(include="number").columns
print("Variables quantitatives :", quanti.tolist())
###2 
# Copier les données pour garder la base originale
df_scores = df.copy()

# Transformer les variables qualitatives
df_scores["Risk"] = df["Risk"].map({
    "High": 0.47,
    "Normal": 0.35,
    "Low": 0.18
})

df_scores["Demand fluctuation"] = df["Demand fluctuation"].map({
    "Increasing": 0.36,
    "Stable": 0.28,
    "Unknown": 0.20,
    "Decreasing": 0.16,
    "Ending": 0.00
})

df_scores["Consignment stock"] = df["Consignment stock"].map({
    "No": 0.80,
    "Yes": 0.20
})

df_scores["Unit size"] = df["Unit size"].map({
    "Large": 0.53,
    "Medium": 0.31,
    "Small": 0.13
})

# Afficher le résultat
print(df_scores.head())
###### cest pour combiner les deux types dans les calculs de l'analyse multi criteres 

###3 
# Normaliser les variables quantitatives entre 0 et 1
colonnes = ["Daily usage", "Average stock", "Lead time", "Unit cost"]
df_norm = df_scores.copy()

for col in colonnes:
    minimum = df_scores[col].min()
    maximum = df_scores[col].max()
    if maximum == minimum:
        df_norm[col] = 0
    else:
        df_norm[col] = (df_scores[col] - minimum) / (maximum - minimum)

# Construire les trois critères agrégés
criteres = pd.DataFrame()

criteres["Criticality"] = (
    0.78 * df_norm["Risk"]
    + 0.22 * df_norm["Demand fluctuation"]
)

criteres["Demand"] = (
    0.71 * df_norm["Daily usage"]
    + 0.29 * df_norm["Average stock"]
)

criteres["Supply"] = (
    0.75 * df_norm["Lead time"]
    + 0.25 * df_norm["Consignment stock"]
)

# Conserver le coût et la taille : cinq critères au total
criteres["Unit cost"] = df_norm["Unit cost"]
criteres["Unit size"] = df_scores["Unit size"]

# Afficher le résultat
print(criteres.head())
## 4
import numpy as np

# 1. Normalisation TOPSIS
normes = np.sqrt((criteres ** 2).sum()).replace(0, 1)
matrice_norm = criteres / normes

# 2. Calcul des poids par entropie
p = criteres / criteres.sum().replace(0, np.nan)
p = p.fillna(1 / len(criteres))

entropie = -(p * np.log(p.where(p > 0, 1))).sum() / np.log(len(criteres))

diversite = (1 - entropie).clip(lower=0)
diversite[criteres.nunique() <= 1] = 0

if diversite.sum() <= 1e-12:
    poids = pd.Series(1 / len(criteres.columns), index=criteres.columns)
else:
    poids = diversite / diversite.sum()

print("Poids par entropie :")
print(poids)

# 3. Matrice pondérée
matrice_ponderee = matrice_norm * poids

# Puis continuer avec les solutions idéales et les distances
ideal_plus = matrice_ponderee.max()
ideal_moins = matrice_ponderee.min()

# Distance à la solution idéale positive et négative
distance_plus = np.sqrt(
    ((matrice_ponderee - ideal_plus) ** 2).sum(axis=1)
)
distance_moins = np.sqrt(
    ((matrice_ponderee - ideal_moins) ** 2).sum(axis=1)
)

# Calculer le score avant de trier les articles en question 5
total_distance = distance_plus + distance_moins
df_scores["Score_TOPSIS"] = (
    distance_moins / total_distance.replace(0, np.nan)
).fillna(0.5)

print(df_scores[["Score_TOPSIS"]].head())
###5 
# Trier du score le plus élevé au plus faible
classement = df_scores.sort_values(
    by="Score_TOPSIS",
    ascending=False,
    kind="mergesort"
).copy()

# Ajouter le rang de chaque article
classement["Rang"] = range(1, len(classement) + 1)

# Afficher les 10 premiers articles
print(classement.head(10))

###6
# Tous les articles sont d'abord en classe C
classement["Classe"] = "C"

# Seuils : 20 % en A, 30 % en B et 50 % en C
seuil_A = int(0.20 * len(df))
seuil_B = int(0.50 * len(df))

# Classe B jusqu'à 50 % des articles
classement.loc[classement["Rang"] <= seuil_B, "Classe"] = "B"

# Classe A : les premiers 20 %
classement.loc[classement["Rang"] <= seuil_A, "Classe"] = "A"

# Afficher le nombre d'articles par classe
print(classement["Classe"].value_counts().sort_index())
### question 7
# Ajouter les classes à la base initiale
# Pandas associe les lignes grâce à leur index d'origine
df["Classe"] = classement["Classe"]

# Enregistrer la base avec la nouvelle colonne
df.to_csv(Path(__file__).parent / "inventory_avec_classes.csv", index=False)

print(df.head())

##"8"
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Les huit variables explicatives, sans le score TOPSIS
X = df_scores.drop(columns=["Score_TOPSIS"])

# La classe à prédire, dans le même ordre que X
y = df["Classe"]

# 70 % pour entraîner et 30 % pour tester
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Définir trois modèles
modeles = {
    "Régression logistique": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Arbre de décision": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),
    "Forêt aléatoire": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Entraîner les modèles et prédire les classes du test
predictions = {}

for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    predictions[nom] = modele.predict(X_test)
    print(nom, ":", predictions[nom][:10])


##9 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

resultats = []

for nom, prediction in predictions.items():
    resultats.append({
        "Modèle": nom,
        "Accuracy": accuracy_score(y_test, prediction),
        "Précision": precision_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "Rappel": recall_score(
            y_test, prediction, average="macro", zero_division=0
        ),
        "F1": f1_score(
            y_test, prediction, average="macro", zero_division=0
        )
    })

    print("\nMatrice de confusion :", nom)
    print(confusion_matrix(
        y_test, prediction, labels=["A", "B", "C"]
    ))

# Comparer les modèles selon le F1
comparaison = pd.DataFrame(resultats)
comparaison = comparaison.sort_values("F1", ascending=False)

print("\nComparaison des modèles :")
print(comparaison.round(3))


