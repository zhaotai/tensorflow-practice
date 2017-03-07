
import tensorflow as tf

STATE = tf.Variable(0, name='counter')
#print STATE.name
ONE = tf.constant(1)

new_value = tf.add(STATE, ONE)
update = tf.assign(STATE, new_value)

init = tf.initialize_all_variables() # must have if define variable

with tf.Session() as SESS:
    SESS.run(init)
    for _ in range(3):
        SESS.run(update)
        print SESS.run(STATE)