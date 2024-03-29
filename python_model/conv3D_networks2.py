import tensorflow as tf
from tensorflow.keras import activations, layers, optimizers
import numpy as np


class FeatureExtractor(tf.keras.Model):

    def __init__(self, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)

        self.conv3d1 = layers.Conv3D(128, 3, strides=1, padding='same', activation='relu')
        self.conv3d2 = layers.Conv3D(64, 3, strides=1, padding='same', activation='relu')
        self.conv3d3 = layers.Conv3D(32, 3, strides=1, padding='same', activation='relu')
        self.conv3d4 = layers.Conv3D(16, 3, strides=1, padding='same', activation='relu')
        self.conv3d5 = layers.Conv3D(8, 3, strides=1, padding='same', activation='relu')
        self.conv1x1 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same')
        self.batchmeup = layers.BatchNormalization(momentum=0.8)
        self.flatten = layers.Flatten()

        """
        self.conv = conv
        self.dense_hidden_units = dense_hidden_units

        if conv:
            self.conv1 = layers.Conv2D(16, kernel_size=8, strides=4, padding="SAME")
            self.conv2 = layers.Conv2D(32, kernel_size=4, strides=2, padding="SAME")

        self.flatten = layers.Flatten()
        if dense_hidden_units > 0:
            self.dense = layers.Dense(dense_hidden_units)
        """

    def call(self, input_shape, sample_action=True):

        #xdim, ydim, zdim, channels = input_shape
        #struct = layers.Input(input_shape)

        x1 = self.conv3d1(input_shape)
        x2 = self.conv3d2(x1)
        x3 = self.conv3d3(x2)
        x4 = self.conv3d4(x3)
        x5 = self.conv3d5(x4)
        xf = tf.concat([x1, x2, x3, x4, x5], -1)
        #xf = tf.squeeze(xf)
        #model = models.Model(inputs=[struct, locked, forces, stayoff], outputs=xfd)
        #return x1,x2,x3,x4
        return xf

class PolicyNetwork(tf.keras.Model):
    """Policy network with discrete action space. Interpretation and encoding of
    actions are not handled here.
    """

    def __init__(self, feature_extractor, **kwargs):
        super(PolicyNetwork, self).__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.conv3d1 = layers.Conv3D(16, 3, strides=(1, 1, 1), padding='same', activation='relu')
        self.conv3d2 = layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', activation='relu')
        self.conv3d3 = layers.Conv3D(64, 3, strides=(1, 1, 1), padding='same', activation='relu')
        self.conv3d4 = layers.Conv3D(32, 3, strides=(1, 1, 1), padding='same', activation='relu')
        self.conv3d5 = layers.Conv3D(16, 3, strides=(1, 1, 1), padding='same', activation='relu')
        """
        self.dense512_1 = layers.Dense(512, activation='relu')
        self.dense512_2 = layers.Dense(512, activation='relu')
        self.dense256_1 = layers.Dense(256, activation='relu')
        self.dense256_2 = layers.Dense(256, activation='relu')
        self.dense128_1 = layers.Dense(128, activation='relu')
        self.dense128_2 = layers.Dense(128, activation='relu')
        self.dense64_1 = layers.Dense(64, activation='relu')
        self.dense32_1 = layers.Dense(32, activation='relu')
        self.dense16_1 = layers.Dense(32, activation='relu')
        """

        self.actions = 3
        self.flatten = layers.Flatten()

    def set_coords(self, struct):
        self.max_dim = tf.math.reduce_max(struct)
        self.dense_coord = layers.Dense(self.actions*self.max_dim, activation='relu')
        #self.dense_coordy = layers.Dense(self.actions*max_dim, activation='relu')
        #self.dense_coordz = layers.Dense(self.actions*max_dim, activation='relu')

    def policy(self, inpu):
        batch, xdim, ydim, zdim, channels = tf.shape(inpu)
        #print(batch, xdim, ydim, zdim, channels)
        #x1,x2 = self.feature_extractor(inpu)
        #x1,x2,x3,x4 = self.feature_extractor(inpu)
        xf = self.feature_extractor(inpu)
        x = self.conv3d1(xf)
        x = self.conv3d2(x)
        x = self.conv3d3(x)
        x = self.conv3d4(x)
        x = self.conv3d5(x)
        """
        x = self.dense128_1(xf)
        x = self.dense256_1(x)
        x = self.dense512_1(x)
        x = self.dense256_2(x)
        x = self.dense128_2(x)
        x = self.dense64_1(x)
        x = self.dense32_1(x)
        x = self.dense16_1(x)
        """
        x = self.flatten(x)

        #x4d = self.conv3d(x4)
        #x4d = self.conv3d(x4d)
        #x4d = self.dense128(x4d)
        #x4d = self.dense128(x4d)

        #xf = tf.concat([x1d, x2d], -1)
        #xf = tf.concat([x1d, x2d, x3d, x4d], -1)

        #xf = self.dense128(xf)
        #xf = self.dense128(xf)
        #xf = self.dense512(xf)
        #xf = self.dense512(xf)
        #xfd = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')(xf)

        #xfd = tf.keras.activations.tanh(xfd)
        #xfd = tf.keras.activations.sigmoid(xfd)
        #xfd = self.dense_coord(x5d)
        #xfd = layers.Reshape((50, zdim))(xfd)
        #xfd = activations.hard_sigmoid(xfd)

        #x = self.dense_coordx(x5d)
        #y = self.dense_coordy(x5d)
        #z = self.dense_coordz(x5d)
        xyz = self.dense_coord(x)
        xyz = layers.Reshape((self.actions, self.max_dim))(xyz)
        #y = layers.Reshape((self.actions, ydim))(y)
        #z = layers.Reshape((self.actions, zdim))(z)
        return xyz

    def _sample_action(self, logits):

        index = tf.random.categorical(logits[0], 1)
        # [batch_size, 1] ==> [batch_size]
        #index = tf.squeeze(index, axis=-1)

        return index

    def call(self, x):

        logits = activations.softmax(self.policy(x))
        action = self._sample_action(logits)

        return action

class ValueNetwork(tf.keras.Model):

    def __init__(self, feature_extractor=None, hidden_units=0, **kwargs):
        super(ValueNetwork, self).__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.hidden_units = hidden_units
        if hidden_units > 0:
            self.hidden = layers.Dense(hidden_units)

        self.value = layers.Dense(1)

    def call(self, observation, time_left):
        #print(tf.shape(observation))
        if self.feature_extractor is not None:
            # TODO: Better to project time_left, so that about same dimensions?
            tr = tf.expand_dims(time_left, -1)
            #print(tr)
            #y0,y1 = self.feature_extractor(observation)
            y0 = self.feature_extractor(observation)
            #feats = tf.concat([y0,y1,y2, y3], -1)
            x = tf.concat([layers.Flatten()(y0), tr], -1)
        else:
            x = time_left

        if self.hidden_units:
            x = self.hidden(x)
            x = activations.relu(x)

        v = tf.squeeze(self.value(x), axis=-1)

        return v
