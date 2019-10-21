# from keras import losses
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks.callbacks import ModelCheckpoint

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
        output_mode="sigmoid"):

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

        self.pool_1, self.mask_1 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_1)

        conv_2 = Convolution2D(128, self.kernel, padding="same")(self.pool_1)
        conv_2 = BatchNormalization()(conv_2)
        self.conv_2 = Activation("relu")(conv_2)

        self.pool_2, self.mask_2 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_2)

        conv_3 = Convolution2D(256, self.kernel, padding="same")(self.pool_2)
        conv_3 = BatchNormalization()(conv_3)
        self.conv_3 = Activation("relu")(conv_3)

        self.pool_3, self.mask_3 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_3)

        conv_4 = Convolution2D(512, self.kernel, padding="same")(self.pool_3)
        conv_4 = BatchNormalization()(conv_4)
        self.conv_4 = Activation("relu")(conv_4)
        conv_5 = Convolution2D(512, self.kernel, padding="same")(self.conv_4)
        conv_5 = BatchNormalization()(conv_5)
        self.conv_5 = Activation("relu")(conv_5)

        self.pool_4, self.mask_4 = MaxPoolingWithArgmax2D(self.pool_size)(self.conv_5)
        print("Done")

        self.encoder_built = True

    def build_decoder(self):

        if self.decoder_built :
            print("Decoder already built.")
            return

        print("Building decoder")

        if not self.encoder_built :
            print("Failed. Encoder not built")
            return

        self.unpool_1 = MaxUnpooling2D(self.pool_size)([self.pool_4, self.mask_4])

        conv_6 = Convolution2D(512, self.kernel, padding="same")(self.unpool_1)
        conv_6 = BatchNormalization()(conv_6)
        self.conv_6 = Activation("relu")(conv_6)

        conv_7 = Convolution2D(512, self.kernel, padding="same")(self.conv_6)
        conv_7 = BatchNormalization()(conv_7)
        self.conv_7 = Activation("relu")(conv_7)

        conv_8 = Convolution2D(256, self.kernel, padding="same")(self.conv_7)
        conv_8 = BatchNormalization()(conv_8)
        self.conv_8 = Activation("relu")(conv_8)

        self.unpool_2 = MaxUnpooling2D(self.pool_size)([self.conv_8,self.mask_3])

        conv_9 = Convolution2D(128, self.kernel, padding="same")(self.unpool_2)
        conv_9 = BatchNormalization()(conv_9)
        self.conv_9 = Activation("relu")(conv_9)

        self.unpool_3 = MaxUnpooling2D(self.pool_size)([self.conv_9,self.mask_2])

        conv_10 = Convolution2D(64, self.kernel, padding="same")(self.unpool_3)
        conv_10 = BatchNormalization()(conv_10)
        self.conv_10 = Activation("relu")(conv_10)

        self.unpool_4 = MaxUnpooling2D(self.pool_size)([self.conv_10,self.mask_1])

        conv_11 = Convolution2D(self.n_classes - 1, self.kernel, padding="same")(self.unpool_4)
        self.conv_11 = BatchNormalization()(conv_11)
        # self.conv_11 = Reshape((-1, self.input_shape[0], self.input_shape[1],self.n_classes-1))(conv_11)
        self.outputs = Activation(self.output_mode)(self.conv_11)
        print("Done.")

    def build_model(self, optimizer, loss, metrics, print_summary = True):

        # INCOMPLETE
        self.build_encoder()
        self.build_decoder()

        self.model = Model(inputs = self.inputs, outputs = self.outputs, name = "SegNet")
        self.model.compile(optimizer= optimizer,loss= loss, metrics = metrics)

        if print_summary :
            print(self.model.summary())


    def train_generator(self, gen_train, steps_per_epoch, epochs, save_path, valid_gen = None, valid_steps = None,
                                    weights_path = None, initial_epoch = 0):

        checkpoint = ModelCheckpoint(save_path, period = 1)
        self.model.fit_generator(gen_train, steps_per_epoch = steps_per_epoch, epochs = epochs,
                                        validation_data = valid_gen, validation_steps = valid_steps,
                                            initial_epoch = initial_epoch, callbacks = [checkpoint])

    def evaluate_generator(self, gen_test, steps, weights_path):

        self.model.load_model(weights_path)
        results = model.evaluate_generator(gen_test, steps)
        print("RESULTS", results)

    def test_image(self, weights_path, image_path, save_path):

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        self.model.load_model(weights_path)

        mask = self.model.predict(np.expand_dims(x, axis = 0))[0]

        masked_img = mask * img
        cv2.imwrite(save_path, masked_img)

