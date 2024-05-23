import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import re
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset():
    try:
        # Crear la raíz de la ventana de Tkinter (pero no mostrarla)
        root = tk.Tk()
        root.withdraw()

        # Abrir una ventana de diálogo para seleccionar el archivo
        file_path = filedialog.askopenfilename(
            title="Seleccione archivo",
            filetypes=(("CSV files", ".csv"), ("all files", ".*"))
        )

        if file_path:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Éxito", "Se cargó el archivo exitosamente!")
            return df
        else:
            messagebox.showwarning("Advertencia", "No se seleccionó ningún archivo.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al cargar el archivo: {e}")
        return None

def mostrar_exploracion(df):
    # Crear una nueva ventana de Tkinter
    window = tk.Tk()
    window.title("Exploración del Dataset")

    # Crear un widget de texto con desplazamiento
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=20)
    text_area.pack(pady=10, padx=10)

    # Obtener resumen de datos
    num_entradas = df.shape[0]
    columnas = df.columns.tolist()
    distribucion_clases = df['Kategori'].value_counts()
    spam_muestras = df[df['Kategori'] == 'spam'].head(2)
    ham_muestras = df[df['Kategori'] == 'ham'].head(2)

    # Agregar la información al widget de texto
    text_area.insert(tk.INSERT, f"Exploración del Dataset\n")
    text_area.insert(tk.INSERT, f"Número de entradas: {num_entradas}\n")
    text_area.insert(tk.INSERT, f"Columnas del dataset: {columnas}\n")
    text_area.insert(tk.INSERT, f"\nDistribución de 'Kategori':\n{distribucion_clases}\n")
    text_area.insert(tk.INSERT, f"\nPrimeras 2 entradas de 'spam':\n{spam_muestras}\n")
    text_area.insert(tk.INSERT, f"\nPrimeras 2 entradas de 'ham':\n{ham_muestras}\n")

    # Crear botón para iniciar el preprocesamiento de datos
    def iniciar_preprocesamiento():
        window.destroy()  # Cerrar la ventana de exploración
        X_train, X_test, y_train, y_test = preprocesar_datos(df)
        if X_train is not None and y_train is not None:
            iniciar_entrenamiento_modelos(X_train, X_test, y_train, y_test)

    boton_preprocesar = tk.Button(window, text="Iniciar Preprocesamiento", command=iniciar_preprocesamiento)
    boton_preprocesar.pack(pady=10)

    # Configurar la ventana para que se cierre con el botón de cerrar
    window.mainloop()

def preprocesar_datos(df):
    # Mostrar mensaje de preprocesamiento en curso
    preprocesamiento_window = tk.Tk()
    preprocesamiento_window.title("Preprocesamiento de Datos")
    preprocesamiento_label = tk.Label(preprocesamiento_window, text="Realizando preprocesamiento de datos, por favor espere...")
    preprocesamiento_label.pack(pady=20, padx=20)
    preprocesamiento_window.update()

    # Realizar el preprocesamiento
    try:
        # Limpieza de datos
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # Codificar la columna 'Kategori' (spam: 1, non-spam: 0)
        le = LabelEncoder()
        df['Kategori'] = le.fit_transform(df['Kategori'])

        # Normalización de texto
        def limpiar_texto(texto):
            texto = texto.lower()
            texto = re.sub(r'\d+', '', texto)  # eliminar números
            texto = re.sub(r'\W+', ' ', texto)  # eliminar caracteres especiales
            return texto

        df['Pesan'] = df['Pesan'].apply(limpiar_texto)

        # Vectorización de texto
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['Pesan'])
        y = df['Kategori']

        # División del conjunto de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Cerrar la ventana de preprocesamiento en curso
        preprocesamiento_window.destroy()

        # Mostrar mensaje de éxito
        messagebox.showinfo("Éxito", "Preprocesamiento realizado con éxito!")

        # Crear una nueva ventana con un botón para iniciar el entrenamiento
        entrenamiento_window = tk.Tk()
        entrenamiento_window.title("Entrenamiento de Modelos")
        entrenamiento_label = tk.Label(entrenamiento_window, text="Haga clic en el botón para iniciar el entrenamiento de modelos.")
        entrenamiento_label.pack(pady=20, padx=20)
        
        def iniciar_entrenamiento():
            entrenamiento_window.destroy()
            entrenar_modelos(X_train, X_test, y_train, y_test)
        
        boton_entrenar = tk.Button(entrenamiento_window, text="Iniciar Entrenamiento", command=iniciar_entrenamiento)
        boton_entrenar.pack(pady=10)

        entrenamiento_window.mainloop()

        return X_train, X_test, y_train, y_test
    except Exception as e:
        preprocesamiento_window.destroy()
        messagebox.showerror("Error", f"Ocurrió un error durante el preprocesamiento: {e}")
        return None, None, None, None

def entrenar_modelos(X_train, X_test, y_train, y_test):
    # Mostrar mensaje de entrenamiento en curso
    entrenamiento_window = tk.Tk()
    entrenamiento_window.title("Entrenamiento de Modelos")
    entrenamiento_label = tk.Label(entrenamiento_window, text="Realizando entrenamiento de modelos, por favor espere...")
    entrenamiento_label.pack(pady=20, padx=20)
    entrenamiento_window.update()

    # Entrenamiento del modelo de Regresión Logística
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    report_logreg = classification_report(y_test, y_pred_logreg)

    # Entrenamiento del modelo K-Vecinos Más Cercanos (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn)

    # Entrenamiento del modelo Red Neuronal (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    report_mlp = classification_report(y_test, y_pred_mlp)

    # Matrices de confusión
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)

    # Cerrar la ventana de entrenamiento en curso
    entrenamiento_window.destroy()

    # Mostrar mensaje de éxito
    messagebox.showinfo("Éxito", "Entrenamiento de modelos completado!")

    # Crear una nueva ventana para mostrar los resultados
    resultados_window = tk.Tk()
    resultados_window.title("Resultados del Entrenamiento")

    # Crear un widget de texto con desplazamiento
    text_area = scrolledtext.ScrolledText(resultados_window, wrap=tk.WORD, width=80, height=20)
    text_area.pack(pady=10, padx=10)

    # Agregar la información al widget de texto
    text_area.insert(tk.INSERT, f"Modelo: Regresión Logística\n")
    text_area.insert(tk.INSERT, f"Precisión: {accuracy_logreg}\n")
    #text_area.insert(tk.INSERT, f"Reporte de Clasificación:\n{report_logreg}\n")
    text_area.insert(tk.INSERT, f"Matriz de Confusión:\n{cm_logreg}\n")

    text_area.insert(tk.INSERT, f"Modelo: K-Vecinos Más Cercanos (KNN)\n")
    text_area.insert(tk.INSERT, f"Precisión: {accuracy_knn}\n")
    #text_area.insert(tk.INSERT, f"Reporte de Clasificación:\n{report_knn}\n")
    text_area.insert(tk.INSERT, f"Matriz de Confusión:\n{cm_knn}\n")

    text_area.insert(tk.INSERT, f"Modelo: Red Neuronal (MLP)\n")
    text_area.insert(tk.INSERT, f"Precisión: {accuracy_mlp}\n")
    #text_area.insert(tk.INSERT, f"Reporte de Clasificación:\n{report_mlp}\n")
    text_area.insert(tk.INSERT, f"Matriz de Confusión:\n{cm_mlp}\n")

    # Configurar la ventana para que se cierre con el botón de cerrar
    resultados_window.mainloop()

def main():
    df = load_dataset()
    if df is not None:
        mostrar_exploracion(df)

if __name__ == "__main__":
    main()