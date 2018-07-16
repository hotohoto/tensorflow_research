import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

inputs = tf.keras.Input(shape=(28, 28,))
flatten_inputs = tf.keras.layers.Flatten()(inputs)
hidden = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flatten_inputs)
dropout = tf.keras.layers.Dropout(0.2)(hidden)
outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)

model.save("num_model.h5")
