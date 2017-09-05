import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Init
random_seed = 0
k = 5  # Number of clusters
n_vectors = 100
np.random.seed(random_seed)

pallete = set(colors.BASE_COLORS.values())
pallete = pallete.union(set([(int(s[1:3], 16) / 255, int(s[3:5], 16) / 255, int(s[5:7], 16) / 255) for s in colors.CSS4_COLORS.values()]))
pallete = [ x for x in pallete if (x[0] + x[1] + x[2]) > (40 / 255)]

assert k < n_vectors

vectors_set = np.random.multivariate_normal([0, 0], [[10, 1], [1, 10]], n_vectors)  # 100x2

vectors = tf.constant(vectors_set)  # 100x2
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors, random_seed), [0, 0], [k, -1]))  # 5x2

expanded_vectors = tf.expand_dims(vectors, 0)  # 1x100x2
expanded_centroids = tf.expand_dims(centroids, 1)  # 5x1x2

# 5x100x2 => 5x100 => 100
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

# 100 => nx1 => n, 100x2 => nx2 => 2 => 5x2
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                     reduction_indices=[1]) for c in range(k)], 0)

update_centroids = tf.assign(centroids, means)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20):
        _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])
        print("Step: ", step)

        for i in range(k):
            plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'bx')
            cluster = [vectors_set[j] for j in range(len(vectors_set)) if assignment_values[j] == i]
            cluster = np.array(cluster)
            plt.plot(cluster[:, 0], cluster[:, 1], 'o', c=pallete[i % len(pallete)])

        plt.show()