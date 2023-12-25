# Lab12:Classification des fleurs Iris
# Réalisé par yahya
# Import des package
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import pandas as pd
import streamlit as st
# Évaluation du modèle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#step1:dataset
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Step2: Model
model = RandomForestClassifier()
# Step3: Train
model.fit(iris.data,iris.target)
# Step4: Test


# Model Deployment on "streamlit run Lab12.py"
st.header('Classification des fleurs Iris')
def user_input():
    sepal_length = st.sidebar.slider("sepal length",4.3,7.9,6.0)
    sepal_width = st.sidebar.slider("sepal width", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("petal length", 1.0, 9.2, 2.0)
    petal_width = st.sidebar.slider("petal width", 0.1, 2.5, 1.0)
    data = {
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features

df = user_input()
st.write(df)
st.subheader("Iris flower Prediction")
prediction = model.predict(df)
# st.write(iris.target_names[prediction])

class_namee = iris.target_names[prediction]
print(class_namee)
class_image_paths = {
    'setosa': 'images/setosa.jpg',  # Remplacez par le chemin de l'image Setosa
    'versicolor': 'images/versicolor.jpg',  # Remplacez par le chemin de l'image Versicolor
    'virginica': 'images/virginica.jpg'  # Remplacez par le chemin de l'image Virginica
}
for class_name, image_path in class_image_paths.items():
    if(class_namee==class_name):
        class_namee=class_name


image_path = class_image_paths.get(class_namee,'images/R.jpg')  # Chemin par défaut si la classe n'est pas trouvée
image = Image.open(image_path)
col1, col2 = st.columns(2)
# Colonne 1 : Afficher le nom de la fleur prédite
with col1:
    st.write(iris.target_names[prediction])

# Colonne 2 : Afficher l'image avec une largeur de 50 pixels
with col2:
    st.image(image, caption=f'Image de la fleur {class_namee}', width=100)
st.image('images/iris.jpeg')

# Prédiction
predictions = model.predict(df_iris)

# Mesures de performance
accuracy = accuracy_score(iris.target, predictions)
precision = precision_score(iris.target, predictions, average='weighted')
recall = recall_score(iris.target, predictions, average='weighted')
f1 = f1_score(iris.target, predictions, average='weighted')
conf_matrix = confusion_matrix(iris.target, predictions)

# Affichage des métriques
st.write(f"Accuracy : {accuracy:.2f}")
st.write(f"Precision : {precision:.2f}")
st.write(f"Recall : {recall:.2f}")
st.write(f"F1 Score : {f1:.2f}")

# Affichage de la matrice de confusion
st.write("Matrice de Confusion :")
st.write(conf_matrix)
# Évaluation du modèle
accuracy = model.score(df_iris, iris.target) * 100
st.write(f"Accuracy : {accuracy:.2f}%")