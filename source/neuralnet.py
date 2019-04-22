import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class ConvNet(object):

    def __init__(self, data_dim, channel, num_class, learning_rate):
