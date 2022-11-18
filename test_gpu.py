import tensorflow as tf
import torch
print(torch.cuda.is_available())
print(tf.config.experimental.list_physical_devices(device_type='GPU'))