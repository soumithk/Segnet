from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D



################################################
# Model from the segnet paper
# https://arxiv.org/pdf/1511.00561.pdf
#
# Extract from the paper
# "The encoder network consists of 13
# convolutional layers which correspond to
# the first 13 convolutional layers in the
# VGG16 network"
#
# SegNet has an encoder network and a
# corresponding decoder network, followed by
# a final pixelwise classification layer.
################################################


class Segnet(object):

    def __init__(self, input_shape, n_classes, kernel = (3, 3), pool_size=(2, 2),
        output_mode="softmax"):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.kernel = kernel
        self.pool_size = pool_size

        self.output_mode = output_mode

        self.encoder_built = False
        self.decoder_built = False
        self.model_built = False
        self.model_compiled = False

    def build_encoder(self):

        if self.encoder_built :
            print("Encoder already built.")
            return

        print("Building encoder")

        self.inputs = Input(shape = self.input_shape)
        conv_1 = Convolution2D(64, self.kernel, padding="same")(self.inputs)
        conv_1 = BatchNormalization()(conv_1)
        self.conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, self.kernel, padding="same")(self.conv_1)
        conv_2 = BatchNormalization()(conv_2)
        self.conv_2 = Activation("relu")(self.conv_2)

        self.pool_1, self.mask_1 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_2)

        conv_3 = Convolution2D(128, self.kernel, padding="same")(self.pool_1)
        conv_3 = BatchNormalization()(conv_3)
        self.conv_3 = Activation("relu")(conv_3)
        conv_4 = Convolution2D(128, self.kernel, padding="same")(self.conv_3)
        conv_4 = BatchNormalization()(conv_4)
        self.conv_4 = Activation("relu")(conv_4)

        self.pool_2, self.mask_2 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_4)

        conv_5 = Convolution2D(256, self.kernel, padding="same")(self.pool_2)
        conv_5 = BatchNormalization()(conv_5)
        self.conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(256, self.kernel, padding="same")(self.conv_5)
        conv_6 = BatchNormalization()(conv_6)
        self.conv_6 = Activation("relu")(conv_6)
        conv_7 = Convolution2D(256, self.kernel, padding="same")(self.conv_6)
        conv_7 = BatchNormalization()(conv_7)
        self.conv_7 = Activation("relu")(conv_7)

        self.pool_3, self.mask_3 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_7)

        conv_8 = Convolution2D(512, self.kernel, padding="same")(self.pool_3)
        conv_8 = BatchNormalization()(conv_8)
        self.conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(512, self.kernel, padding="same")(self.conv_8)
        conv_9 = BatchNormalization()(conv_9)
        self.conv_9 = Activation("relu")(conv_9)
        conv_10 = Convolution2D(512, self.kernel, padding="same")(self.conv_9)
        conv_10 = BatchNormalization()(conv_10)
        self.conv_10 = Activation("relu")(conv_10)

        self.pool_4, self.mask_4 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_10)

        conv_11 = Convolution2D(512, self.kernel, padding="same")(self.pool_4)
        conv_11 = BatchNormalization()(conv_11)
        self.conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(512, self.kernel, padding="same")(self.conv_11)
        conv_12 = BatchNormalization()(conv_12)
        self.conv_12 = Activation("relu")(conv_12)
        conv_13 = Convolution2D(512, self.kernel, padding="same")(self.conv_12)
        conv_13 = BatchNormalization()(conv_13)
        self.conv_13 = Activation("relu")(conv_13)

        self.pool_5, self.mask_5 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_13)
        print("Done")

        self.encoder_built = True

    def build_decoder(self):

        if self.decoder_built :
            print("Decoder already built.")
            return

        print("Building decoder")

        if not decoder_built :
            print("Failed. Encoder not built")
            return

        self.unpool_1 = MaxUnpooling2D(pool_size)([self.pool_5,self. mask_5])

        conv_14 = Convolution2D(512, self.kernel, padding="same")(self.unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        self.conv_14 = Activation("relu")(conv_14)
        conv_15 = Convolution2D(512, self.kernel, padding="same")(self.conv_14)
        conv_15 = BatchNormalization()(conv_15)
        self.conv_15 = Activation("relu")(conv_15)
        conv_16 = Convolution2D(512, self.kernel, padding="same")(self.conv_15)
        conv_16 = BatchNormalization()(conv_16)
        self.conv_16 = Activation("relu")(conv_16)

        self.unpool_2 = MaxUnpooling2D(pool_size)([self.conv_16, self.mask_4])

        conv_17 = Convolution2D(512, self.kernel, padding="same")(self.unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        self.conv_17 = Activation("relu")(conv_17)
        conv_18 = Convolution2D(512, self.kernel, padding="same")(self.conv_17)
        conv_18 = BatchNormalization()(conv_18)
        self.conv_18 = Activation("relu")(conv_18)
        conv_19 = Convolution2D(256, self.kernel, padding="same")(self.conv_18)
        conv_19 = BatchNormalization()(conv_19)
        self.conv_19 = Activation("relu")(conv_19)

        self.unpool_3 = MaxUnpooling2D(pool_size)([self.conv_19, self.mask_3])

        conv_20 = Convolution2D(256, self.kernel, padding="same")(self.unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        self.conv_20 = Activation("relu")(conv_20)
        conv_21 = Convolution2D(256, self.kernel, padding="same")(self.conv_20)
        conv_21 = BatchNormalization()(conv_21)
        self.conv_21 = Activation("relu")(conv_21)
        conv_22 = Convolution2D(128, self.kernel, padding="same")(self.conv_21)
        conv_22 = BatchNormalization()(conv_22)
        self.conv_22 = Activation("relu")(conv_22)

        self.unpool_4 = MaxUnpooling2D(pool_size)([self.conv_22, self.mask_2])

        conv_23 = Convolution2D(128, self.kernel, padding="same")(self.unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        self.conv_23 = Activation("relu")(conv_23)
        conv_24 = Convolution2D(64, self.kernel, padding="same")(self.conv_23)
        conv_24 = BatchNormalization()(conv_24)
        self.conv_24 = Activation("relu")(conv_24)

        self.unpool_5 = MaxUnpooling2D(pool_size)([self.conv_24, self.mask_1])

        conv_25 = Convolution2D(64, self.kernel, padding="same")(self.unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        self.conv_25 = Activation("relu")(conv_25)

        conv_26 = Convolution2D(self.n_classes, (1, 1), padding="valid")(self.conv_25)
        conv_26 = BatchNormalization()(conv_26)
        self.conv_26 = Reshape(
                (self.input_shape[0]*self.input_shape[1], self.n_classes),
                input_shape=(self.input_shape[0], self.input_shape[1], self.n_classes))(conv_26)

        self.outputs = Activation(self.output_mode)(self.conv_26)
        print("Done.")

    def build_model(self):

        # INCOMPLETE
        self.build_encoder()
        self.build_decoder()

        self.model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    def train_generator(self, gen_train, steps_per_epoch, epochs, valid_gen = None, valid_steps = None,
                                    weights_path = None, initial_epoch = 0):

        # TODO - add callbacks model checkpoint
        self.model.fit_generator(gen_train, steps_per_epoch = steps_per_epoch, epochs = epochs,
                                        validation_data = valid_gen, validation_steps = valid_steps,
                                            initial_epoch = initial_epoch)
