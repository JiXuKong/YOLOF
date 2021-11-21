import tensorflow as tf

slim = tf.contrib.slim

def act(x):
    return tf.nn.relu6(x)

def conv2d(input, channel, kernel, dilation=1, reuse=False, scope=None, bias = None, is_training=True):
        output = slim.conv2d(input, channel, kernel,
                                    weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    rate = dilation,
                                    padding="SAME",
                                    biases_initializer=bias,
                                    activation_fn=None,
                                    reuse = reuse,
                                    trainable=is_training,
                                    scope=scope)
        return output


def bn_(input_, esp=1e-3, is_training = True, decay = 0.99, scope = 'bn'):
    x = tf.layers.batch_normalization(
        inputs = input_,
        axis=-1,
        name = scope,
        momentum= 0.997,
        epsilon= 1e-4,
        training= is_training)
#         fused=True)
    return x