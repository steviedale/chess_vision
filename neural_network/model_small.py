import tensorflow as tf


class ModelSmall:
    @staticmethod
    def create_model(input_shape, num_classes, activation="relu", weightsPath=None):
        # initialize the model
        model = tf.keras.Sequential()

        #if we are using "channels first", update the input shape
        if tf.keras.backend.image_data_format() == "channels_first":
            input_shape = (input_shape[2], input_shape[0], input_shape[1])

        #layer-1
        model.add(tf.keras.layers.Conv2D(50, (5, 5),
										 activation=activation,
                                         input_shape=input_shape, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))  # 256x256 --> 64x64

        # layer-2
        model.add(tf.keras.layers.Conv2D(25, (3, 3),
                                         activation=activation,
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))  # 64x64 --> 16x16

        # layer-3
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(200, activation=activation, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        # define the third and last FC layer
        model.add(tf.keras.layers.Dense(num_classes))

        # lastly, define the soft-max classifier
        model.add(tf.keras.layers.Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
