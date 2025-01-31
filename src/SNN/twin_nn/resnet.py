import numpy as np

# ResNet model
import tensorflow.keras as keras
import tensorflow as tf
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import utils


class ResNet:
    
    def __init__(self, input_shape=None):
    
        self.model = self.build_model(input_shape)
        # FIXME: save summary in file self.model.summary()
        
        return
    
    
    def build_model(self, input_shape):
        
        n_feature_maps = 64
        
        input_layer = keras.layers.Input(input_shape)
        
        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=2, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=2, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=2, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        
        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        
        # BLOCK 3
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=2, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        
        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)
        
        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        
        
        # BLOCK 4
        #conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        #conv_x = keras.layers.BatchNormalization()(conv_x)
        #conv_x = keras.layers.Activation('relu')(conv_x)
        
        #conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        #conv_y = keras.layers.BatchNormalization()(conv_y)
        #conv_y = keras.layers.Activation('relu')(conv_y)
        
        #conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        #conv_z = keras.layers.BatchNormalization()(conv_z)
        
        # no need to expand channels because they are equal
        #shortcut_y = keras.layers.BatchNormalization()(output_block_3)
        
        #output_block_4 = keras.layers.add([shortcut_y, conv_z])
        #output_block_4 = keras.layers.Activation('relu')(output_block_4)
        
        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        #dense_layer = keras.layers.Dense(32, activation='relu')(gap_layer)
        
        embedding_network = keras.Model(inputs=input_layer, outputs=gap_layer)
        #embedding_network = keras.Model(inputs=input_layer, outputs=dense_layer)
      
        return embedding_network

    
    def get_model(self):
        
        return self.model
