import tensorflow as tf
from tensorflow.keras import activations, layers, optimizers



class FeatureExtractor(tf.keras.Model):

    def __init__(self, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)

        self.conv3d1 = layers.Conv3D(16, 5, strides=3, padding='same', activation='relu')
        self.conv3d2 = layers.Conv3D(16, 5, strides=3, padding='same', activation='relu')
        self.conv3d3 = layers.Conv3D(16, 5, strides=3, padding='same', activation='relu')
        self.conv3d4 = layers.Conv3D(16, 5, strides=3, padding='same', activation='relu')
        self.conv1x1 = layers.Conv3D(1, 1, strides=(1, 1, 1), padding='same', activation='relu')
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
        x1 = self.conv1x1(x1)
        #x1 = self.conv3d1(x1)
        #x1 = self.batchmeup(x1)
        
        x2 = self.conv3d2(x1)
        x2 = self.conv1x1(x2)
        #x2 = self.conv3d2(x2)
        #x2 = self.batchmeup(x2)
        
        x3 = self.conv3d3(x2)
        x3 = self.conv1x1(x3)
        #x3 = self.conv3d3(x3)
        #x3 = self.batchmeup(x3)
        
        x4 = self.conv3d4(x3)
        x4 = self.conv1x1(x4)
        #x4 = self.batchmeup(x4)
        
        xf1 = self.flatten(x1)
        xf2 = self.flatten(x2)
        xf3 = self.flatten(x3)
        xf4 = self.flatten(x4)
        print(tf.shape(xf1))
        print(tf.shape(xf2))
        print(tf.shape(xf3))
        print(tf.shape(xf4))
        xf = tf.concat([xf1, xf2, xf3, xf4], -1)
        #xf = tf.squeeze(xf)
        print(tf.shape(xf))
        #model = models.Model(inputs=[struct, locked, forces, stayoff], outputs=xfd)

        return xf

class PolicyNetwork(tf.keras.Model):
    """Policy network with discrete action space. Interpretation and encoding of
    actions are not handled here.
    """

    def __init__(self, feature_extractor, **kwargs):
        super(PolicyNetwork, self).__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.conv3d = layers.Conv3D(32, 5, strides=(1, 1, 1), padding='same', activation='relu')
        self.dense512 = layers.Dense(512)
        self.dense256 = layers.Dense(256)
        self.dense128 = layers.Dense(128)
        self.dense64 = layers.Dense(64)
        self.dense32 = layers.Dense(32)
        self.dense16 = layers.Dense(16)

    def policy(self, inpu):
        batch, xdim, ydim, zdim, channels = tf.shape(inpu)
        #print(batch, xdim, ydim, zdim, channels)
        #x1,x2 = self.feature_extractor(inpu)
        #x1,x2,x3,x4 = self.feature_extractor(inpu)
        xf = self.feature_extractor(inpu)
        #print(tf.shape(x1))
        #x1d = self.conv3d(x1)
        #x1d = self.conv3d(x1d)
        x1d = self.dense128(xf)
        print(tf.shape(x1d))
        #x1d = self.dense128(x1d)

        #x2d = self.conv3d(x2)
        #x2d = self.conv3d(x2d)
        x2d = layers.Dense(256)(x1d)

        #x3d = self.conv3d(x3)
        #x3d = self.conv3d(x3d)
        x3d = layers.Dense(256)(x2d)

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
        x4d = layers.Dense(150)(x3d)
        xfd = layers.Reshape((50, 3))(x4d) 

        return xfd

    def _sample_action(self, logits):

        xfd = layers.Reshape((xdim, ydim, zdim))(xfd) 
        return xfd

    def call(self, x):

        logits = self.policy(x)
        #action = self._sample_action(logits)

        return logits

class ValueNetwork(tf.keras.Model):

    def __init__(self, feature_extractor=None, hidden_units=32, **kwargs):
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
            x = tf.concat([feats, tr], -1)
        else:
            x = time_left

        if self.hidden_units:
            x = self.hidden(x)
            x = activations.relu(x)

        v = tf.squeeze(self.value(x), axis=-1)

        return v