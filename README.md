# StockLab — Devoir libre

Application Streamlit pour la classification ABC de 700 articles de stock.

## Installation et lancement

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pour exécuter les questions 1 à 9 en ligne de commande :

```powershell
python solution_devoir.py
```

## Méthode

- Identification et conversion des huit attributs.
- Construction de cinq critères : Criticality, Demand, Supply, Unit cost et Unit size.
- Pondération par entropie, puis classement TOPSIS.
- Classes ABC selon le nombre d'articles : 20 % A, 30 % B et 50 % C.
- Comparaison de la régression logistique, de l'arbre de décision et de la forêt aléatoire.
- Export CSV des résultats. L'approche floue est exclue.

Les modèles apprennent à reproduire les classes TOPSIS. Les résultats de test ne constituent pas une validation métier des priorités.

L'application utilise uniquement le fichier inventory_data.csv du projet.
Le script génère inventory_avec_classes.csv.

Référence : Kartal et al. (2016), DOI 10.1016/j.cie.2016.06.004.

