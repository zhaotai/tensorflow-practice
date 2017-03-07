""" tensorflow tutorial2: matrix multiply """
import tensorflow as tf
MATRIX1 = tf.constant([[3, 3]])
MATRIX2 = tf.constant([[2], [2]])
PRODUCT = tf.matmul(MATRIX1, MATRIX2)   # matrix multiply

# method 1
# sess = tf.Session()
#result = sess.run(PRODUCT)
#print(result)
#sess.close()

# method 2
with tf.Session() as sess:
    RESULT2 = sess.run(PRODUCT)
    print RESULT2
