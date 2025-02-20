import tensorflow as tf

conc_yearly_unet_model = tf.keras.models.load_model(f'/mnt/dsk1/yhlee/workdir/cmaqnet/models/cond_unet_o3_all')
conc_yearly_unet_model.summary()
print(conc_yearly_unet_model.loss)