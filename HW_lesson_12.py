import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(__file__)

TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH, image_size=(128, 128), batch_size = 32,
    label_mode = 'categorical'
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH, image_size=(128, 128), batch_size = 32,
    label_mode = 'categorical'
)


model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(128, 128, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

#шар який аналізує ознаки:
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#навчання моделі
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(train_ds, epochs=10, validation_data=test_ds)

test_photo = os.path.join(BASE_DIR, 'images', '125.jpg')

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# class_name = ["apple", "bikes", "plane"]
class_name = train_ds.class_names

if os.path.exists(test_photo):
# class_name = sorted(os.listdir(TRAIN_PATH))
# img = image.load_img(test_photo, target_size=(128, 128))
    img = image.load_img("images/234.jpg", target_size=(128, 128))

    img_array = image.img_to_array(img)
    # img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    result_index = np.argmax(predictions[0])

    print(f'Результат: {class_name[result_index]}')
    print("Ймовірності по класах:", predictions[0])