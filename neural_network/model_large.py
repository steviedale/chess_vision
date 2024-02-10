import tensorflow as tf


class ModelLarge:
    @staticmethod
    def create_model(input_shape, num_classes, activation="relu", weightsPath=None):
        # initialize the model
        model = tf.keras.Sequential()

        #if we are using "channels first", update the input shape
        if tf.keras.backend.image_data_format() == "channels_first":
            input_shape = (input_shape[2], input_shape[0], input_shape[1])

        # layer-1
        model.add(tf.keras.layers.Conv2D(50, (5, 5), activation=activation, 
                                         input_shape=input_shape, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 256x256 --> 128x128

        # layer-2
        model.add(tf.keras.layers.Conv2D(100, (3, 3),
                                         activation=activation,
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 128x128 --> 64x64

        # layer-3
        model.add(tf.keras.layers.Conv2D(150, (3, 3),
                                         activation=activation,
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 64x64 --> 32x32 

        # layer-4
        model.add(tf.keras.layers.Conv2D(200, (3, 3),
                                         activation=activation,
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 32x32 --> 16x16 

        # layer-5
        model.add(tf.keras.layers.Conv2D(250, (3, 3),
                                         activation=activation,
                                         use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 16x16 --> 8x8

        # layer-6
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1000, activation=activation, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        # layer-7
        model.add(tf.keras.layers.Dense(400, activation=activation, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())

        # define the third and last FC layer
        model.add(tf.keras.layers.Dense(num_classes))

        # lastly, define the soft-max classifier
        model.add(tf.keras.layers.Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
