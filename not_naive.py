# Importing the libraries
import tensorflow as tf
import tensorflow_datasets as tdfs
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop

# Model definition
from tensorflow.keras import layers, Model, Input

input_layer = Input(shape=(100, 100, 3), name="img")
x = layers.Conv2D(16, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(32,(3,3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64,(3,3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64,(3,3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64,(3,3), activation="relu")(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
output_layer = layers.Dense(1, activation="sigmoid")(x)

model = Model(input_layer, output_layer, name="CNN")
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# EXTRACTION PHASE
data = tdfs.load("malaria", split='train', as_supervised=True)

# Reading the saved files with the raw data
file_pattern = f'~/tensorflow_datasets/malaria/1.0.0/malaria-train.tfrecord*'
file = tf.data.Dataset.list_files(file_pattern)

# Using the interleave function to concurrently load the files
train_dataset = file.interleave(
    tf.data.TFRecordDataset,
    cycle_length=4, # Number of files to read in parallel
    num_parallel_calls=tf.data.experimental.AUTOTUNE # Number of CPU threads to use for parallel processing
)
# EXTRACTION PHASE END

# TRANSFORMATION PHASE
def augmentationV2(serialized_data):
    description ={
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64, -1)
    }
    example = tf.io.parse_single_example(serialized_data, description)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (100, 100))
    image = image / 255
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, example['label']

import multiprocessing
# Number of CPU cores available
cores = multiprocessing.cpu_count()

# Using the map function to apply the augmentation function to the dataset in parallel on the CPU cores
train_dataset = train_dataset.map(augmentationV2, num_parallel_calls=cores)

# Cache the dataset to memory to avoid reading from disk every epoch
train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(100).batch(32)
# Prefetching the data to improve the speed of the training
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# TRANSROFMATION PHASE END


# LOAD PHASE
history = model.fit(
    train_dataset,
    epochs=15
)
# LOAD PHASE END