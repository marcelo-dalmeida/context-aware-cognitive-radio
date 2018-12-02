import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def load(self, data_path, session, ignore_missing=False):
    '''Load network weights.
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    ignore_missing: If true, serialized weights for missing layers are ignored.
    '''
    data_dict = np.load(data_path).item()
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for param_name, data in data_dict[op_name].iteritems():
                try:
                    var = tf.get_variable(param_name)
                    session.run(var.assign(data))
                except ValueError:
                    if not ignore_missing:
                        raise

def cnn_to_mlp(input, convs, hiddens, scope="any", max_poolings=None, layer_norm=None, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride, use_max_pooling, use_layer_norm in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

                if use_max_pooling and max_poolings:
                    pool_size, strides, padding = max_poolings.pop(0)
                    out = tf.layers.max_pooling2d(out,
                                                  pool_size=pool_size,
                                                  strides=strides,
                                                  padding=padding)

                if use_layer_norm and layer_norm:
                    radius, alpha, beta = layer_norm.pop(0)
                    out = tf.nn.local_response_normalization(input,
                                                              depth_radius=radius,
                                                              alpha=alpha,
                                                              beta=beta)

        conv_out = layers.flatten(out)

        dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                  shape=[],
                                                  name='use_dropout')
        with tf.variable_scope("class_value"):
            out = conv_out
            for hidden, use_dropout in hiddens:
                out = layers.fully_connected(out,
                                             num_outputs=hidden)

                if use_dropout:
                    keep = 1 - dropout + (dropout * 0.01)
                    out = tf.nn.dropout(out, keep)

            class_probabilities = tf.nn.softmax(out)

        return class_probabilities
