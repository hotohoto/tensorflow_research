import tensorflow as tf

x = tf.Variable(0.0)

y = (x-2)**2

step = tf.train.AdamOptimizer(0.01).minimize(y)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    _x, _y = sess.run([x, y])
    print(0, _x, _y)
    for i in range(10000):
        _, _x, _y = sess.run([step, x, y])
        if (i + 1) % 100 == 0:
            print(i + 1, _x, _y)
