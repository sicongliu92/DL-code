import tensorflow as tf
import numpy as np

a = tf.ones(shape=[1,6,6,1],dtype=tf.float32)

pool = tf.nn.max_pool(a, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')


def unpool2(pool, ksize, stride, padding = 'VALID'):
    """
    simple unpool method

    :param pool : the tensor to run unpool operation
    :param ksize : integer
    :param stride : integer
    :return : the tensor after the unpool operation

    """
    pool = tf.transpose(pool, perm=[0,3,1,2])
    pool_shape = pool.shape.as_list()
    if padding == 'VALID':
        size = (pool_shape[2] - 1) * stride + ksize
    else:
        size = pool_shape[2] * stride
    unpool_shape = [pool_shape[0], pool_shape[1], size, size]
    unpool = tf.Variable(np.zeros(unpool_shape), dtype=tf.float32)
    for batch in range(pool_shape[0]):
        for channel in range(pool_shape[1]):
            for w in range(pool_shape[2]):
                for h in range(pool_shape[3]):
                    diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[batch,channel,w*stride,h*stride]],values=tf.expand_dims(pool[batch][channel][w][h],axis=0),dense_shape = [pool_shape[0],pool_shape[1],size,size]))
                    unpool = unpool + diff_matrix
    
    unpool = tf.transpose(unpool, perm=[0,2,3,1])
    return unpool
