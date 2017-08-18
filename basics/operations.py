import tensorflow as tf

print("\n\nStep 1")
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a+b=%i" % sess.run(a + b))
    print("a*b=%i" % sess.run(a * b))


print("\n\nStep 2")
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("a+b=%i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("a*b=%i" % sess.run(mul, feed_dict={a: 2, b: 3}))

print("\n\nStep 3")

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print("a*b =", result)