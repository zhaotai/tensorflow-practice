"""tutorial1: basic structure"""
import tensorflow as tf
import numpy as np

# create data
X_DATA = np.random.rand(100).astype(np.float32)
Y_DATA = X_DATA * 0.1 + 0.3

### crete tensorflow structure start ###
WEIGHTS = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
BIASES = tf.Variable(tf.zeros([1]))

Y = WEIGHTS * X_DATA + BIASES

LOSS = tf.reduce_mean(tf.square(Y - Y_DATA))
OPTIMIZER = tf.train.GradientDescentOptimizer(0.5)
TRAIN = OPTIMIZER.minimize(LOSS)

INIT = tf.global_variables_initializer()
### crete tensorflow structure end ###

SESS = tf.Session()
SESS.run(INIT)      #Very important

for step in range(201):
    SESS.run(TRAIN)
    if step % 20 == 0:
        print(step, SESS.run(WEIGHTS), SESS.run(BIASES))
