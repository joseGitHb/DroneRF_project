#%%
import pandas as pd
from createDataBase import createDataBase
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
import joblib
#%%
rootPath = r"E:\DataBase\MPACT_DroneRC_RF_Dataset"
numSignals=150
start_file = 600
opts = 'IncludeFeatures'
tableFormat=True
#%%
dataBase0_150 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase0_150.to_csv(r"C:\Users\joset\DroneRF_project\dataBase0_150.csv", index=False)
#%%
data0_150 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase0_150.csv")
#%%
dataBase150_300 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase150_300.to_csv(r"C:\Users\joset\DroneRF_project\dataBase150_300.csv", index=False)
#%%
data150_300 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase150_300.csv")
#%%
dataBase300_450 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase300_450.to_csv(r"C:\Users\joset\DroneRF_project\dataBase300_450.csv", index=False)
#%%
data300_450 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase300_450.csv")
#%%
dataBase450_600 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase450_600.to_csv(r"C:\Users\joset\DroneRF_project\dataBase450_600.csv", index=False)
#%%
data450_600 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase450_600.csv")
#%%
dataBase600_700 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase600_700.to_csv(r"C:\Users\joset\DroneRF_project\dataBase600_700.csv", index=False)
#%%
data600_700 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase600_700.csv")
#%%
data = pd.concat([data0_150, data150_300, data300_450, data450_600, data600_700], ignore_index=True)
#%%
data["make_model"]=data.loc[:, "make"]+"_"+data.loc[:,"model"]
#%%
features = data.loc[:, "Mean":]
features = features.drop("Entropy", axis=1)
#%%
train_set_full, test_set = train_test_split(features, test_size=0.2, random_state=42)
train_set, valid_set = train_test_split(train_set_full, test_size=0.2, random_state=42)
# %%
# Separamos datos y etiquetas 
x_train = train_set.drop("make_model", axis=1)
y_train = train_set["make_model"].copy()
x_valid = valid_set.drop("make_model", axis=1)
y_valid = valid_set["make_model"].copy()
#%% Feature Scaling
std_scaler = StandardScaler() # Para usar Gradient Descent
x_train_std_scaler = std_scaler.fit_transform(x_train)
df_x_train_std_scaler = pd.DataFrame(x_train_std_scaler, columns=x_train.columns, 
                                      index=x_train.index)
x_valid_std_scaler = std_scaler.transform(x_valid)
df_x_valid_std_scaler = pd.DataFrame(x_valid_std_scaler, columns=x_valid.columns, 
                                      index=x_valid.index)

#%% Select model
#%% Bayesian Optimization for a Keras model
#%% 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train_encod = encoder.fit_transform(y_train)
y_valid_encod = encoder.transform(y_valid)
#%% EarlySptopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
#%% Keras Tuner
import tensorflow as tf
import keras_tuner as kt

def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=(13,)))
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(hp.Float(f"l2_{i}", 1e-5, 1e-2, sampling="log"))
        ))
    model.add(keras.layers.Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))
    model.add(keras.layers.Dense(17, activation="softmax"))
    # Tune the learning rate for the optimizer.
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model

tuner = kt.BayesianOptimization(
    build_model, objective="val_accuracy", max_trials=35, directory="keras_tuner_dir",
    project_name="tuner_BH_3")

tuner.search(df_x_train_std_scaler, y_train_encod, epochs=50,
             validation_data=(df_x_valid_std_scaler, y_valid_encod),
             callbacks=[early_stopping_cb])
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for hp in best_hps.values:
    print(f"{hp}: {best_hps.get(hp)}")


#%%
history = best_model.fit(df_x_train_std_scaler, y_train_encod, 
                    epochs=100, validation_data=(df_x_valid_std_scaler, y_valid_encod),
                    callbacks=[early_stopping_cb])
#%% Plot learning curves
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 100], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")  # extra code
plt.savefig("keras_learning_curves_plot_BH")  # extra code

#%% Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
y_valid_pred = best_model.predict(df_x_valid_std_scaler).argmax(axis=1)
fig, ax = plt.subplots(figsize=(12,12))
ConfusionMatrixDisplay.from_predictions(y_valid_encod, y_valid_pred, normalize="true",
                                        values_format=".0%", ax=ax, xticks_rotation=45)
plt.savefig("ConfusionMatrixKeras_BH", dpi=300, bbox_inches="tight")

# %% Other metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Macro average (treats all classes equally)
print("Macro Precision:", precision_score(y_valid_encod, y_valid_pred, average="macro"))
print("Macro Recall:", recall_score(y_valid_encod, y_valid_pred, average="macro"))
print("Macro F1-score:", f1_score(y_valid_encod, y_valid_pred, average="macro"))

# Weighted average (accounts for class imbalance)
print("Weighted Precision:", precision_score(y_valid_encod, y_valid_pred, average="weighted"))
print("Weighted Recall:", recall_score(y_valid_encod, y_valid_pred, average="weighted"))
print("Weighted F1-score:", f1_score(y_valid_encod, y_valid_pred, average="weighted"))

# Full report
print(classification_report(y_valid_encod, y_valid_pred, digits=4))

#%% Evaluate on Test Set
x_test = test_set.drop("make_model", axis=1)
x_test_std_scaler = std_scaler.transform(x_test)
df_x_test_std_scaler = pd.DataFrame(x_test_std_scaler, columns=x_test.columns, 
                                      index=x_test.index)
y_test = test_set["make_model"].copy()
y_test_encod = encoder.transform(y_test)

test_loss, test_accuracy = best_model.evaluate(df_x_test_std_scaler, y_test_encod)
print(f"Test accuracy: {test_accuracy:.4f}")    
print(f"Test loss: {test_loss:.4f}")

#%% 95% confidence interval

#%% Save the model
best_model.save("keras_best_model.h5")

#Load the model
#model = keras.models.load_model("keras_model.h5")



#%% MULTICLASS CLASSIFICATION
# Scickit-Learn ejecuta OvR or OvO según el algoritmo cuando se quiere 
# ejecutar clasificación binaria para una tarea de clasificación multiclase
from sklearn.svm import SVC 
"""
    Hiperparámetros: C
"""
svm_clf = SVC(random_state=42)
svm_clf.fit(df_x_train_std_scaler, y_train)

#svm_clf.predict([df_x_train_std_scaler.loc[1000]])
some_instance_scores = svm_clf.decision_function([df_x_train_std_scaler.iloc[300]])
some_instance_scores.round(2)
# %%
svm_clf.classes_
# %% ERROR ANALYSIS
from sklearn.metrics import ConfusionMatrixDisplay
y_train_pred = cross_val_predict(svm_clf, df_x_train_std_scaler, y_train, cv=3)
#%% Matriz de confusión
fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=ax)
plt.savefig("ConfusionMatrixSVM" , dpi=300, bbox_inches="tight")
#%% Matriz de confusión normalizada
fig, ax = plt.subplots(figsize=(12,12))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, 
                                        normalize="true", values_format=".0%",
                                         ax=ax, xticks_rotation=45)
plt.savefig("ConfusionMatrixSVM_normalize", dpi=300, bbox_inches="tight")

# %% Le asignamos peso 0 a las predicciones correctas para ver mejor los errores
sample_weight = (y_train_pred != y_train)
fig, ax = plt.subplots(figsize=(12,12))
plt.rc('font', size=10)  # extra code
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%",
                                        ax=ax, xticks_rotation=45)
plt.savefig("Errors normalized by row", dpi=300, bbox_inches="tight")
# %%
