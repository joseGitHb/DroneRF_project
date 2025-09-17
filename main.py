# main.py
import pandas as pd
from createDataBase import createDataBase
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
import keras
import keras_tuner as kt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score

rootPath = r"E:\DataBase\MPACT_DroneRC_RF_Dataset" # Directorio raíz de la base de datos
numSignals=150 # Número de señales a cargar por modelo (150, 150, 150, 150, 100)
start_file = 0 # Índice del primer archivo a cargar (0, 150, 300, 450, 600)
opts = 'IncludeFeatures' # 'OnlyRawData', 'OnlyCroppedData', 'IncludeFeatures'
tableFormat=True # Devuelve un DataFrame si es True, si es False devuelve una lista de objetos DronRC

# Crear cinco bases de datos parciales y guardarlas en archivos CSV
dataBase0_150 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase0_150.to_csv(r"C:\Users\joset\DroneRF_project\dataBase0_150.csv", index=False)

dataBase150_300 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase150_300.to_csv(r"C:\Users\joset\DroneRF_project\dataBase150_300.csv", index=False)

# Se hace lo mismo para dataBase300_450.csv, dataBase450_600.csv y dataBase600_700.csv
dataBase300_450 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase300_450.to_csv(r"C:\Users\joset\DroneRF_project\dataBase300_450.csv", index=False)

dataBase450_600 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase450_600.to_csv(r"C:\Users\joset\DroneRF_project\dataBase450_600.csv", index=False)

dataBase600_700 = createDataBase(rootPath, numSignals, opts, tableFormat, start_file)
dataBase600_700.to_csv(r"C:\Users\joset\DroneRF_project\dataBase600_700.csv", index=False)

# Cargar las cinco bases de datos parciales 
data0_150 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase0_150.csv")
data150_300 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase150_300.csv")
data300_450 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase300_450.csv")
data450_600 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase450_600.csv")
data600_700 = pd.read_csv(r"C:\Users\joset\DroneRF_project\dataBase600_700.csv")
# Combinar las cinco bases de datos en una sola
data = pd.concat([data0_150, data150_300, data300_450, data450_600, data600_700], ignore_index=True)

data["make_model"]=data.loc[:, "make"]+"_"+data.loc[:,"model"] # Generación de etiquetas
features = data.loc[:, "Mean":] # Sólo las columnas de características estadísticas
features = features.drop("Entropy", axis=1) # Eliminamos Entropy por tener valores infinitos

# Dividimos en train, validation y test sets (60%, 20%, 20%)
train_set_full, test_set = train_test_split(features, test_size=0.2, random_state=42)
train_set, valid_set = train_test_split(train_set_full, test_size=0.2, random_state=42)

# Separamos datos (x) y etiquetas (y) 
x_train = train_set.drop("make_model", axis=1)
y_train = train_set["make_model"].copy()

x_valid = valid_set.drop("make_model", axis=1)
y_valid = valid_set["make_model"].copy()

x_test = test_set.drop("make_model", axis=1)
y_test = test_set["make_model"].copy()

#Feature Scaling
std_scaler = StandardScaler() 
# Entrenar sólo con el conjunto de entrenamiento. Luego se aplica a valid y test
x_train_std_scaler = std_scaler.fit_transform(x_train)
# Convertir el array numpy devuelto por fit_transform a DataFrame de pandas
df_x_train_std_scaler = pd.DataFrame(x_train_std_scaler, columns=x_train.columns, 
                                      index=x_train.index)
# Transformar valid y crear DataFrame
x_valid_std_scaler = std_scaler.transform(x_valid)
df_x_valid_std_scaler = pd.DataFrame(x_valid_std_scaler, columns=x_valid.columns, 
                                      index=x_valid.index)
# Transformar test y crear DataFrame
x_test_std_scaler = std_scaler.transform(x_test)
df_x_test_std_scaler = pd.DataFrame(x_test_std_scaler, columns=x_test.columns, 
                                      index=x_test.index)

# Los modelos Keras esperan que las etiquetas sean enteros
encoder = LabelEncoder()
y_train_encod = encoder.fit_transform(y_train)
y_valid_encod = encoder.transform(y_valid)
# Early Stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Keras Tuner
# Definimos la función build_model(hp) que crea el modelo Keras y define el espacio de búsqueda de hiperparámetros.
# Esta función será utilizada por Keras Tuner para explorar diferentes configuraciones de la red neuronal.
def build_model(hp):
    model = keras.models.Sequential() # Modelo secuencial
    model.add(keras.layers.InputLayer(shape=(13,)))  # Capa de entrada con 13 características
    # Bucle para añadir capas ocultas, el número de capas se ajusta como hiperparámetro.
    for i in range(hp.Int("num_layers", 1, 3)): # Entre 1 y 3 capas ocultas
        model.add(keras.layers.Dense(
            units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16), # Unidades por capa entre 16 y 128
            activation="relu", # Función de activación ReLU
            kernel_regularizer=keras.regularizers.l2(hp.Float(f"l2_{i}", 1e-5, 1e-2, sampling="log")) # Regularización L2
        ))
    # Añade una capa de Dropout para regularización, el ratio se ajusta como hiperparámetro
    model.add(keras.layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)))
    # Capa de salida con 17 clases y activación softmax
    model.add(keras.layers.Dense(17, activation="softmax"))
    # Ajusta la tasa de aprendizaje del optimizador Adam como hiperparámetro
    model.compile(
        optimizer=keras.optimizers.Adam( 
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")), # Tasa de aprendizaje entre 0.0001 y 0.01
        loss="sparse_categorical_crossentropy", # Pérdida para clasificación multiclase
        metrics=["accuracy"]) # Métrica de precisión
    return model 
# Instanciamos el tuner de Keras Tuner usando optimización bayesiana (con prior de proceso gaussiano)
tuner = kt.BayesianOptimization(
    build_model, # función de construcción del modelo
    objective="val_accuracy", # métrica objetivo a maximizar
    max_trials=35, # número máximo de combinaciones de hiperparámetros a probar
    directory="keras_tuner_dir",  # carpeta donde se guardan los resultados
    project_name="tuner_BH_4") # nombre del proyecto para organizar los resultados

# Ejecutamos la búsqueda de hiperparámetros usando el tuner
# Entrenamos cada modelo candidato con los datos de entrenamiento
# Evaluamos accuracy con los datos de validación
tuner.search(df_x_train_std_scaler, y_train_encod, epochs=50,
             validation_data=(df_x_valid_std_scaler, y_valid_encod),
             callbacks=[early_stopping_cb]) # Early stopping para evitar sobreentrenamiento


# Obtenemos el mejor modelo encontrado por el tuner
best_model = tuner.get_best_models(num_models=1)[0]
# Obtenemos los mejores hiperparámetros encontrados
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
for hp in best_hps.values:
    print(f"{hp}: {best_hps.get(hp)}")

# Entrenamos el mejor modelo usando early stopping
history = best_model.fit(df_x_train_std_scaler, y_train_encod, 
                    epochs=100, validation_data=(df_x_valid_std_scaler, y_valid_encod),
                    callbacks=[early_stopping_cb])


#Generar curvas de aprendizaje
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 13],
    ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="center left")  
plt.savefig("keras_learning_curves_plot_BH_13") 

# Evualuamos el mejor modelo con el conjunto de test
y_test_encod = encoder.transform(y_test)
test_loss, test_accuracy = best_model.evaluate(df_x_test_std_scaler, y_test_encod)


# Predicciones del conjunto de validación. Argmax para obtener la clase predicha
y_valid_pred = best_model.predict(df_x_valid_std_scaler).argmax(axis=1)  
# Matriz de confusión normalizada por filas.
cm = confusion_matrix(y_valid_encod, y_valid_pred, normalize="true")
fig, ax = plt.subplots(figsize=(12,12))
# Para visualizar nombres en las etiquetas de los ejes, se usa display_labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=encoder.classes_) 
disp.plot(values_format=".0%", ax=ax, xticks_rotation=-90)
plt.savefig("ConfusionMatrixKeras_BH", dpi=300, bbox_inches="tight")

# Precision, Recall, F1-score 
print("Precision:", precision_score(y_valid_encod, y_valid_pred, average="macro"))
print("Recall:", recall_score(y_valid_encod, y_valid_pred, average="macro"))
print("F1-score:", f1_score(y_valid_encod, y_valid_pred, average="macro"))


# Save the model
best_model.save("keras_best_model.h5")

#Load the model
#best_model = keras.models.load_model("keras_model.h5")

