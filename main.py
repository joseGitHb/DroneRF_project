#%%
import pandas as pd
from createDataBase import createDataBase
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
#%%
rootPath = r"E:\DataBase\MPACT_DroneRC_RF_Dataset"
numSignals=30
opts = 'IncludeFeatures'
tableFormat=True
dataBase = createDataBase(rootPath, numSignals, opts, tableFormat)
dataBase.to_csv(r"C:\Users\joset\DroneRF_project\dataBase.csv", index=False)
#%%
data = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase300.csv")

#%%
features = data.loc[:, "Mean":]
#%%
train_set, test_set = train_test_split(features, test_size=0.2, random_state=42)
# %%
# Separamos datos y etiquetas 
x_train = train_set.drop("make_model", axis=1)
y_train = train_set["make_model"].copy()
#%% Feature Scaling
std_scaler = StandardScaler()
x_train_std_scaler = std_scaler.fit_transform(x_train)
df_x_train_std_scaler = pd.DataFrame(x_train_std_scaler, columns=x_train.columns, 
                                      index=x_train.index)

#%% Select model

#%% Cross Validation

#%% RandomizedSearchCV or HalvingRandomSearchCV 

#%% Evaluate on Test Set
x_test = test_set.drop("make_model", axis=1)
y_test = test_set["make_model"].copy()

#%% 95% confidence interval

#%% Save the model
#joblib.dump(final_model, "final_model.pkl")



#%% Primero probamos un clasificador binario: "FlySky_FST6 classifier"
# Distingue entre 2 clases: "FlySky_FST6" & "non FlySky_FST6"

# Creamos el vector de etiquetas 
y_train_FlySky_FST6 = (y_train == 'FlySky_FST6')
y_test_FlySky_FST6 = (y_test == 'FlySky_FST6')
# %% SGD Classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(df_x_train_std_scaler, y_train_FlySky_FST6)

# %%
sgd_clf.predict([df_x_train_std_scaler.loc[1130]])
# %% Measuring Accuracy using CrossValidation
cross_val_score(sgd_clf, df_x_train_std_scaler, y_train_FlySky_FST6, 
                cv=3, scoring="accuracy")

# %% Dummy classifier
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(df_x_train_std_scaler, y_train_FlySky_FST6)
dummy_clf.predict(df_x_train_std_scaler)

# %%
cross_val_score(dummy_clf, df_x_train_std_scaler, y_train_FlySky_FST6, 
                cv=3, scoring="accuracy")
#%% Matriz de confusión. Mejor para evaluar el modelo
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_FlySky_FST6_pred = cross_val_predict(sgd_clf, df_x_train_std_scaler, 
                                             y_train_FlySky_FST6)
cm = confusion_matrix(y_train_FlySky_FST6, y_train_FlySky_FST6_pred)
#%% Precision & Recall 
from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_FlySky_FST6, y_train_FlySky_FST6_pred))
#%% Determinamos el umbral de decisión
y_scores = cross_val_predict(sgd_clf, df_x_train_std_scaler, y_train_FlySky_FST6, 
                             cv=3, method="decision_function")
#%% Curva PR. Mejor cuando la clase negativa aparece poco
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_FlySky_FST6, y_scores)

plt.figure(figsize=(8, 4)) 
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
threshold = 0
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-3, 3, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
plt.savefig("precision_recall_vs_threshold.png")

# %%
import matplotlib.patches as patches  # extra code – for the curved arrow
plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
# extra code – just beautifies and saves Figure 3–6
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko",
         label="Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.savefig("precision_recall_curve.png")
#%% ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_FlySky_FST6, y_scores)
plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.savefig("ROC_Curve")
#%%  Compute Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_FlySky_FST6, y_scores)

#%% Ahora creamos un RandomForestClassifier para comparar
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, df_x_train_std_scaler, y_train_FlySky_FST6, 
                                    cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_FlySky_FST6, y_scores_forest)
plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

# extra code – just beautifies and saves Figure 3–8
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.savefig("pr_curve_comparison_plot")
#%% MULTICLASS CLASSIFICATION
# Scickit-Learn ejecuta OvR or OvO según el algoritmo cuando se quiere 
# ejecutar clasificación binaria para una tarea de clasificación multiclase
from sklearn.svm import SVC
svm_clf = SVC(random_state=42)
svm_clf.fit(df_x_train_std_scaler, y_train)
# %%
#svm_clf.predict([df_x_train_std_scaler.loc[1000]])
some_instance_scores = svm_clf.decision_function([df_x_train_std_scaler.loc[1000]])
some_instance_scores.round(2)
# %%
svm_clf.classes_
# %% ERROR ANALYSIS
from sklearn.metrics import ConfusionMatrixDisplay
y_train_pred = cross_val_predict(svm_clf, df_x_train_std_scaler, y_train, cv=3)
#%%
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.savefig("ConfusionMatrixSVM")
#%%
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.savefig("ConfusionMatrixSVM_normalize")

# %%
