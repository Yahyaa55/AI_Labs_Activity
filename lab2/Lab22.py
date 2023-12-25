# lab21 : Classification des legumes et des fruits
# Realisation oar : yahya Emsi 2023/2024
# REf : dataset  !wget https://bitbucket.org/ishaanjav/code-and-deploy-custom-tensorflow-lite-model/raw/a4febbfee178324b2083e322cdead7465d6fdf95/fruits.zip
# REf : codesource  https://colab.research.google.com/drive/1mpehJKSeQyL9I8GAQh53pHmNCtAFKr8F?hl=fr#scrollTo=yYCcRf6jumBp

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

## Step 1: Dataset
#Data manipulation
img_height, img_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)
#Data visualisation
class_names = ["tofaha", "banana", "limouna"]
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
## Step 2: Model
model = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1./255),
     tf.keras.layers.Conv2D(32, 3, activation="relu"), # CNN   # 3: kernel
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Conv2D(32, 3, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(3)
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)


##Step 3: Train
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)
##Step 4: Test
model.evaluate(test_ds)
plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1):
  classifications = model(images)

  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    index = numpy.argmax(classifications[i])
    plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])
plt.show()

##Model deployment with Streamlit
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", 'wb') as f:
  f.write(tflite_model)