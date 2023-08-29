import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

current_dir = os.getcwd()
data_path = os.path.join(current_dir, "data/mnist.npz")
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path) 

def reshape_and_normalize(images):
    new_images = np.expand_dims(images, axis=3)
    new_images = new_images / 255.0
    return new_images

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
      print("\nReached 99.5% accuracy so cancelling training!") 
      self.model.stop_training = True

# Convolutional model
def convolutional_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model
    
training_images = reshape_and_normalize(training_images)

model = convolutional_model()

callbacks = myCallback()

# This can take a little while
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])