import keras as k


class Cnn:

    INPUT_SHAPE = (512, None, 1)
    ACTIVATION_FUNCTION = "relu"
    PADDING = "valid"
    OUTPUT_SIZE = 1251
    LEARNING_RATE = 0.01  # set this to 0.01 * (0.85 ** n) where n is current epoch
    LEARNING_RATE_DECAY = 0.85  # as in original matlab config
    L2_WEIGHT_LOSS = 5e-4  # as above
    MOMENTUM = 0.9  # as above
    EPOCHS = 30

    def __init__(self,
                 weight_path: str = None,
                 use_batch_norm: bool = True,
                 output_size: int = OUTPUT_SIZE):
        self.weight_path = weight_path
        self.use_batch_norm = use_batch_norm
        self.output_size = output_size
        self.model = None

    def _compile(self):
        """Compiles classifier with SGD optimizer"""
        opt = k.optimizers.SGD(lr=Cnn.LEARNING_RATE, momentum=Cnn.MOMENTUM)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy",
                           metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    def _build(self):
        """Builds network graph"""

        def conv_layer(
                prev,
                filters,
                kernel_size,
                stride,
                padding,
                id,
                use_batch_norm=self.use_batch_norm,
                activation=Cnn.ACTIVATION_FUNCTION):

            prev = k.layers.ZeroPadding2D(
                padding=padding)(prev)
            prev = k.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding=Cnn.PADDING,
                kernel_regularizer=k.regularizers.l2(Cnn.L2_WEIGHT_LOSS),
                name=id)(prev)
            if use_batch_norm:
                prev = k.layers.BatchNormalization(name="bnorm" + str(id))(prev)
            return k.layers.Activation(activation)(prev)

        def pooling(x, size, stride, id):
            id = str(id)
            return k.layers.MaxPooling2D(
                pool_size=size,
                strides=stride,
                name="MaxPool" + id)(x)

        def global_pooling(x, reshaped_size, id):
            id = str(id)
            x = k.layers.GlobalAveragePooling2D(name="GlobalAvgPool" + id)(x)
            return k.layers.Reshape(
                reshaped_size,
                name="Reshape" + id)(x)

        model_input = k.Input(Cnn.INPUT_SHAPE, name="ModelInput")
        model_output = conv_layer(model_input, filters=96, kernel_size=(7, 7),
                       stride=(2, 2), padding=(1, 1), id="conv1")
        model_output = pooling(model_output, size=(3, 3), stride=(2, 2), id=1)

        model_output = conv_layer(model_output, filters=256, kernel_size=(5, 5),
                       stride=(2, 2), padding=(1, 1), id="conv2")
        model_output = pooling(model_output, size=(3, 3), stride=(2, 2), id=2)

        model_output = conv_layer(model_output, filters=384, kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1), id="conv3")
        model_output = conv_layer(model_output, filters=256, kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1), id="conv4")
        model_output = conv_layer(model_output, filters=256, kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1), id="conv5")
        model_output = pooling(model_output, size=(5, 3), stride=(3, 2), id=5)

        model_output = conv_layer(model_output, filters=4096, kernel_size=(9, 1),
                       stride=(1, 1), padding=(0, 0), id="fc6")
        model_output = global_pooling(model_output, reshaped_size=(1, 1, 4096), id=6)

        model_output = conv_layer(model_output, filters=1024, kernel_size=(1, 1),
                       stride=(1, 1), padding=(0, 0), id="fc7")
        model_output = k.layers.Conv2D(filters=self.output_size, kernel_size=(1, 1),
                       strides=(1, 1), padding="valid", kernel_regularizer=k.regularizers.l2(Cnn.L2_WEIGHT_LOSS),
                       name="fc8")(model_output)
        model_output = k.layers.Flatten()(model_output)
        model_output = k.layers.Activation("softmax")(model_output)

        self.model = k.Model(model_input, model_output)
        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)

    @staticmethod
    def _schedule_learning_rate(epoch, lr):
        """Decreases learning rate by decay"""
        if epoch > 0:
            return lr * Cnn.LEARNING_RATE_DECAY
        return lr

    def fit_generator(self,
                      model_checkpoint_path,
                      train_generator,
                      train_steps,
                      validation_generator,
                      validation_steps,
                      epochs=EPOCHS):
        self._build()
        self._compile()
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[
                k.callbacks.ModelCheckpoint(
                    model_checkpoint_path,
                    save_best_only=True),
                k.callbacks.LearningRateScheduler(
                    Cnn._schedule_learning_rate,
                    verbose=1)])

    def evaluate_generator(self, test_generator, test_steps):
        self._build()
        self._compile()
        return self.model.evaluate_generator(test_generator, test_steps, verbose=1)
