# Importing the libraries
import tensorflow as tf
import tensorflow_datasets as tdfs
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# EXTRACTION PHASE
data = tdfs.load("malaria", split='train', as_supervised=True)
# EXTRACTION PHASE END

# TRANSFORMATION PHASE
def augmentation(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (100, 100))
    image = image / 255
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

train = data.map(augmentation)
train_batches = train.shuffle(100).batch(32)
# TRANSFORMATION PHASE END

# LOAD PHASE
history = model.fit(
    train_batches,
    epochs=15
)
# LOAD PHASE END