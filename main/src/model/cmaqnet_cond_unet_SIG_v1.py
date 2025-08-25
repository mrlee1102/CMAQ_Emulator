from typing import Sequence
import tensorflow as tf

from src.model.layers_SIG_v1 import GriddingLayer
from src.utils.params import HEIGHT, WIDTH

def embedding_1d(tensor, emb_dim:int, activation:str):
    """Embedding layer for 1D input tensor."""
    emb = tf.keras.layers.Dense(emb_dim)(tensor)
    emb = tf.keras.layers.Activation(activation)(emb)
    return emb

def time_embedding_1d(tensor, emb_dim:int, activation:str):
    """Time embedding layer for 1D input tensor."""
    half_dim = emb_dim // 2
    emb = tf.math.log(10000.) / (half_dim - 1)
    emb = tf.math.exp(-tf.range(half_dim, delta=emb))
    emb = tensor * emb[None, :]
    emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
    emb = tf.keras.layers.Dense(emb_dim)(emb)
    emb = tf.keras.layers.Activation(activation)(emb)
    return emb

def encoder_block(x, embs:list, filters:int, kernel:int, activation:str, dropout:float):
    """Encoder block for conditional U-Net."""
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x_params = []
    for emb in embs:
        emb = tf.keras.layers.Dense(filters)(emb)
        emb = tf.keras.layers.Activation(activation)(emb)
        emb = tf.keras.layers.Reshape((1, 1, filters))(emb)
        emb = tf.keras.layers.Multiply()([x, emb])
        x_params.append(emb)
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.Add()([x] + x_params)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def decoder_block(x, xi, embs:list, filters:int, kernel:int, activation:str, dropout:float):
    """Decoder block for conditional U-Net."""
    x = tf.keras.layers.Conv2DTranspose(filters, kernel, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x_params = []
    for emb in embs:
        emb = tf.keras.layers.Dense(filters)(emb)
        emb = tf.keras.layers.Activation(activation)(emb)
        emb = tf.keras.layers.Reshape((1, 1, filters))(emb)
        emb = tf.keras.layers.Multiply()([x, emb])
        x_params.append(emb)
    x = tf.keras.layers.Concatenate()([x, xi])
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same')(x)
    x = tf.keras.layers.Add()([x] + x_params)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def build_model(
    ctrl_dim:int,
    cond_dim:Sequence[int],
    emb_dims:Sequence[int],
    emb_type:Sequence[str],
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
        cond_dim (Sequence[int]): dimension of each conditional input
        emb_dims (Sequence[int]): dimension of each embedding layer
        emb_type (Sequence[str]): type of each embedding layer (time or normal)
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
    cond_num = len(cond_dim)
    if cond_num != len(emb_dims) or cond_num != len(emb_type):
        raise ValueError('Length of cond_dim, emb_dims, and emb_type must be the same.')
    
    # define input layers
    inputs = [tf.keras.Input(shape=(ctrl_dim,), name='ctrl_input')]

    for i in range(cond_num):
        inputs.append(tf.keras.Input(shape=(cond_dim[i],), name=f'cond_input_{i}'))
    
    # define init layers
    ctrl_maps = GriddingLayer()(inputs[0], ctrl_dim=ctrl_dim//17)
    ctrl_maps = tf.keras.layers.Resizing(*hidden_size)(ctrl_maps)
    

    embs = []
    for i in range(cond_num):
        if emb_type[i] == 'time':
            embs.append(time_embedding_1d(inputs[i+1], emb_dims[i], activation))
        else:
            embs.append(embedding_1d(inputs[i+1], emb_dims[i], activation))
    
    # define encoder blocks
    x = x0 = encoder_block(ctrl_maps, embs, in_filters, kernel_size, activation, dropout)
    xs = [x0]
    for i in range(1, hidden_depth):
        x = xi = encoder_block(x, embs, in_filters*2**i, kernel_size, activation, dropout)
        x = tf.keras.layers.MaxPool2D()(x)
        xs.append(xi)
    
    # define bottleneck layer
    x = encoder_block(x, embs, in_filters*2**hidden_depth, kernel_size, activation, dropout)

    # define decoder blocks
    for i in range(hidden_depth-1):
        x = decoder_block(x, xs[-i-1], embs, in_filters*2**(hidden_depth-i-1), kernel_size, activation, dropout)

    # define output layers
    x = tf.keras.layers.Resizing(HEIGHT, WIDTH)(x)
    outputs = tf.keras.layers.Conv2D(out_channel, 1)(x)
    if use_abs:
        outputs = tf.abs(outputs)
        

    return tf.keras.Model(inputs=inputs, outputs=outputs)