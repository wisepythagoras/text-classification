from keras import backend as K
import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.layers import Dense, Activation, Multiply, Add, Lambda
from keras import initializers
# from tf.keras.initializers import Constant


# This came from here: https://github.com/ParikhKadam/Highway-Layer-Keras/blob/master/highway_layer.py
class Highway(Layer):
    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.transform_gate_bias is None:
            raise Exception('transform_gate_bias is none')

        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = tf.initializers.Constant(self.transform_gate_bias)
        input_shape_dense_1 = input_shape[-1]
        layer_name = self.name if self.name is not None else '' + ':'

        self.dense_1 = Dense(
            input_shape=input_shape,
            units=dim,
            bias_initializer=transform_gate_bias_initializer,
            name=layer_name + 'internal_dense_1'
        )
        self.dense_2 = Dense(
            input_shape=input_shape,
            units=dim,
            name=layer_name + 'internal_dense_2'
        )
        self._trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        shape = K.int_shape(x)

        if shape is None:
            raise Exception('Shape is none')

        dim = shape[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation('sigmoid', name='highway_activation')(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,), name='carry_gate_lambda')(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation, name='transformed_activation')(transformed_data)
        transformed_gated = Multiply(name='transformed_multiply')([transform_gate, transformed_data])
        identity_gated = Multiply(name='identity_gated')([carry_gate, x])
        value = Add(name='highway_value_add')([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config
