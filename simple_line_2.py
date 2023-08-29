import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Train model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=24)
    
    return model

model = house_model()

new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)