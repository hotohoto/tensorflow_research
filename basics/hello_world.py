import tensorflow as tf

message = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run(message))