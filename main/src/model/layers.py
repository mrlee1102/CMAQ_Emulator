import pandas as pd
import tensorflow as tf
from src.utils.alloc import allocation
alloc_df = pd.DataFrame(
    allocation,
    columns=['Region_Name', 'Region_Code', 'Row', 'Column', 'Ratio']
)

@tf.keras.utils.register_keras_serializable(package='CMAQNet')
class GriddingLayer(tf.keras.layers.Layer):
    region_codes = {
        'A': 'Seoul City', 'B': 'Incheon City', 'C': 'Busan City', 'D': 'Daegu City',
        'E': 'Gwangju City', 'F': 'Gyeonggi-do', 'G': 'Gangwon-do', 'H': 'Chungbuk-do',
        'I': 'Chungnam-do', 'J': 'Gyeongbuk-do', 'K': 'Gyeongnam-do', 'L': 'Jeonbuk-do',
        'M': 'Jeonnam-do', 'N': 'Jeju-do', 'O': 'Daejeon City', 'P': 'Ulsan City', 'Q': 'Sejong City'}
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check if allocation is a dictionary and convert it to DataFrame
        if isinstance(allocation, dict):
            # Convert dictionary back to DataFrame
            self.allocation = pd.DataFrame(alloc_df)
        else:
            self.allocation = alloc_df

        # Pre-compute indices and ratios for each region
        self.indices = []
        self.ratios = []
        for key in self.region_codes.keys():
            index = self.allocation.loc[self.allocation['Region_Code'] == key, ['Row', 'Column']].values
            ratio = self.allocation.loc[self.allocation['Region_Code'] == key, ['Ratio']].values / 100
            self.indices.append(tf.constant(index, dtype=tf.int32))
            self.ratios.append(tf.constant(ratio, shape=(1, len(ratio), 1), dtype=tf.float32))
    
    from typing import Optional
    def call(self, inputs, ctrl_dim:Optional[int] = None):
        if ctrl_dim is None:
            raise ValueError("The 'ctrl_dim' parameter must be specified when calling GriddingLayer.")
        # reshaped_inputs = tf.reshape(inputs, (-1, 17, ctrl_dim))
        # batch_size = tf.shape(inputs)[0]

        batch_size = tf.shape(inputs)[0]
        reshaped_inputs = tf.reshape(inputs, tf.stack([batch_size, 17, ctrl_dim]))
        ctrl_map_shape_batch = (batch_size, 82, 67, ctrl_dim)
        ctrl_map = tf.zeros(ctrl_map_shape_batch, dtype=tf.float32)
        for i, (index, ratio) in enumerate(zip(self.indices, self.ratios)):
            value = reshaped_inputs[:, i:i+1, :] * ratio
            batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, tf.shape(index)[0]])
            full_indices = tf.concat([batch_indices[..., tf.newaxis], tf.tile(index[tf.newaxis, ...], [batch_size, 1, 1])], axis=-1)
            ctrl_map = tf.tensor_scatter_nd_add(ctrl_map, full_indices, value)
        return ctrl_map

    def get_config(self):
        # Serialize allocation as a dictionary for model saving
        config = super().get_config()
        config.update({
            "region_codes": self.region_codes,
            "allocation": self.allocation.to_dict(orient='list')})
        return config
    
    @classmethod
    def from_config(cls, config):
        # 'allocation'을 DataFrame으로 변환하여 복원
        config['allocation'] = pd.DataFrame(config['allocation'])
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='CMAQNet')
class RegriddingLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape=(82, 67, 1), **kwargs):
        super().__init__(**kwargs)
        if isinstance(allocation, dict):
            self.allocation = pd.DataFrame(alloc_df)
        else:
            self.allocation = alloc_df

        # Extract row and column indices from allocation
        self.row_indices = tf.constant(self.allocation['Row'].values, dtype=tf.int32)
        self.col_indices = tf.constant(self.allocation['Column'].values, dtype=tf.int32)
        self.target_shape = target_shape

    def call(self, inputs):
        # Start with a zeroed tensor of the target shape
        batch_size = tf.shape(inputs)[0]
        regrid_map = tf.zeros((batch_size,) + self.target_shape, dtype=inputs.dtype)

        # Create a complete set of indices for each batch item
        batch_indices = tf.range(batch_size) # 256
        batch_indices = tf.repeat(batch_indices, repeats=[tf.shape(self.row_indices)[0]] * batch_size)
        # [0..0, 0..0, 0..0, 1..1, 1..1, 1..1, ... 255..255, 255..255, 255..255]
        # Prepare complete indices for the scatter operation
        full_row_indices = tf.tile(self.row_indices, [batch_size])  # Repeat row indices for all batches
        full_col_indices = tf.tile(self.col_indices, [batch_size])  # Repeat col indices for all batches
        scatter_indices = tf.stack([batch_indices, full_row_indices, full_col_indices], axis=1)

        # Flatten inputs to be scattered
        flat_inputs = tf.reshape(inputs, [-1])  # Flatten all inputs for scattering

        print(scatter_indices.shape, inputs.shape, regrid_map.shape, flat_inputs.shape)

        # Scatter the flattened inputs into the zeroed map at the scatter indices
        regrid_map = tf.tensor_scatter_nd_update(regrid_map, scatter_indices, flat_inputs)

        return regrid_map

    def get_config(self):
        config = super().get_config()
        config.update({
            "allocation": self.allocation.to_dict(orient='list'),
            "target_shape": self.target_shape
        })
        return config
    
    def get_config(self):
        # Serialize allocation as a dictionary for model saving
        config = super().get_config()
        config.update({
            "region_codes": self.region_codes,
            "allocation": self.allocation.to_dict(orient='list')})
        return config