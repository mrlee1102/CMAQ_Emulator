from typing import Sequence
import tensorflow as tf

from src.model.layers import GriddingLayer
from src.utils.params import HEIGHT, WIDTH

def encoder_block(x, filters:int, kernel:int, activation:str, dropout:float):
    """Encoder block for conditional U-Net."""
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def decoder_block(x, xi, filters:int, kernel:int, activation:str, dropout:float):
    """Decoder block for conditional U-Net."""
    x = tf.keras.layers.Conv2DTranspose(filters, kernel, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Concatenate()([x, xi])
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def build_model(
    ctrl_dim:int,
    out_channel:int,
    hidden_size:Sequence[int],
    hidden_depth:int,
    in_filters:int,
    kernel_size:int,
    activation:str,
    dropout:float,
    use_abs:bool) -> tf.keras.Model:
    """Build conditional U-Net tensorflow model.

    Args:
        ctrl_dim (int): total number (dimension) of control matrix
        out_channel (int): number of output channels (number of pollutants)
        hidden_size (Sequence[int]): size of resized hidden layer
        hidden_depth (int): depth of U-Net model
        in_filters (int): number of initial filters
        kernel_size (int): kernel size of convolutional layers
        activation (str): activation function
        dropout (float): dropout rate, 0.0 for no dropout
        use_abs (bool): whether to use absolute value for output

    Returns:
        tf.keras.Model: conditional U-Net model
    """
    # define input layers
    inputs = [tf.keras.Input(shape=(ctrl_dim,))]
    
    # define init layers
    ctrl_maps = GriddingLayer()(inputs[0], ctrl_dim//17)
    ctrl_maps = tf.keras.layers.Resizing(*hidden_size)(ctrl_maps)
    
    # define encoder blocks
    x = x0 = encoder_block(ctrl_maps, in_filters, kernel_size, activation, dropout)
    xs = [x0]
    for i in range(1, hidden_depth):
        x = xi = encoder_block(x, in_filters*2**i, kernel_size, activation, dropout)
        x = tf.keras.layers.MaxPool2D()(x)
        xs.append(xi)
    
    # define bottleneck layer
    x = encoder_block(x, in_filters*2**hidden_depth, kernel_size, activation, dropout)

    # define decoder blocks
    for i in range(hidden_depth-1):
        x = decoder_block(x, xs[-i-1], in_filters*2**(hidden_depth-i-1), kernel_size, activation, dropout)

    # define output layers
    x = tf.keras.layers.Resizing(HEIGHT, WIDTH)(x)
    outputs = tf.keras.layers.Conv2D(out_channel, 1)(x)
    if use_abs:
        outputs = tf.abs(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)