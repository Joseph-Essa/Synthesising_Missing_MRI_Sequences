from tkinter import Tcl
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, LeakyReLU, Activation, Concatenate, GlobalAveragePooling2D, MaxPool2D, Reshape, Dense, Multiply, UpSampling2D

# Generator model
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create trainable parameters gamma and beta
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Compute the mean and variance along the spatial dimensions
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def conv_block(x, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = Activation("relu")(x)

    return x


def se_block(x, num_filters, ratio=8):
    se_shape = (1, 1, num_filters)
    se = GlobalAveragePooling2D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(num_filters // ratio, activation="relu", use_bias=False)(se)
    se = Dense(num_filters, activation="sigmoid", use_bias=False)(se)
    se = Reshape(se_shape)(se)
    x = Multiply()([x, se])
    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    x = se_block(x, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(x, s, num_filters):
    x = UpSampling2D(interpolation="bilinear")(x)
    x = Concatenate()([x, s])
    x = conv_block(x, num_filters)
    x = se_block(x, num_filters)
    return x

def squeeze_attention_unet(input_shape=(256, 256, 3)):
    """ Inputs """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)


    b1 = conv_block(p4, 1024)
    b1 = se_block(b1, 1024)
    


    """ Decoder """
    d =  decoder_block(b1, s4, 512)
    d1 = decoder_block(d, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    """ Outputs """
    outputs = Conv2D(3, (1, 1), activation='tanh')(d3)

    """ Model """
    
    model = Model(inputs, outputs, name="Squeeze-Attention-UNET")
    return model

generator_g = squeeze_attention_unet()
