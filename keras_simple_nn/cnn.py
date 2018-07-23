import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

inputs = tf.keras.Input(shape=(28, 28))
reshaped = tf.keras.layers.Reshape((28, 28, 1), input_shape=(None, 28, 28))(inputs)
conv1 = tf.keras.layers.Conv2D(kernel_size=(5,5), filters=32, activation='relu')(reshaped)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2))(conv1)
conv2 = tf.keras.layers.Conv2D(kernel_size=(5,5), filters=64, activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2))(conv2)
flatten_inputs = tf.keras.layers.Flatten()(pool2)
dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(flatten_inputs)
dropout = tf.keras.layers.Dropout(0.4)(dense1)
dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
model = tf.keras.Model(inputs=inputs, outputs=dense2)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)

model.save("num_model.h5")
