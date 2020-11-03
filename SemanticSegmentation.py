#Dataset
#http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np
import cv2
from glob import glob
from time import time



IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 151
batch_size = 8


def createConvolutionBlock(inputs, filters, pool=False, dropout=False, dilation=False):
    initializer = 'he_normal'


    if dilation:
        if filters>=32:
            x = Conv2D(filters, 2, padding="same",dilation_rate=(1,1),kernel_initializer=initializer)(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, 3, padding="same",dilation_rate=(2,2),kernel_initializer=initializer)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, 3, padding="same",dilation_rate=(3,3),kernel_initializer=initializer)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        else:
            x = Conv2D(filters, 2, padding="same",dilation_rate=(1,1),kernel_initializer=initializer)(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, 3, padding="same",dilation_rate=(2,2),kernel_initializer=initializer)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        
        if pool:
            pl = MaxPooling2D((2,2))(x)
            if dropout:
                x = Dropout(0.50)(x)
            return x, pl
        else:
            if dropout:
                x = Dropout(0.50)(x)
            return x

    else:
        y = Conv2D(filters, 2, padding="same",kernel_initializer=initializer)(inputs)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(filters, 3, padding="same",kernel_initializer=initializer)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        
        if pool:
            pool_ = MaxPooling2D((2,2))(y)
            if dropout:
                y = Dropout(0.50)(y)
            return y, pool_
        else:
            if dropout:
                y = Dropout(0.50)(y)
            return y

def createDecoderBlock(inputlayer, concatlayer, filters, dropout=False, dilation=False):
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(inputlayer) # "nearest", "bilinear", "bicubic"
    actv = Activation("relu")(u1)
    c1 = Concatenate()([actv, concatlayer])
    conv_block2 = createConvolutionBlock(c1, filters, pool=False, dropout=dropout,dilation=dilation)

    return conv_block2




def build_model_architecture(shape, num_classes):
    inputs = Input(shape)

    actv2, pl2 = createConvolutionBlock(inputs, 32, pool=True, dropout=False, dilation=True)
    actv3, pl3 = createConvolutionBlock(pl2, 64, pool=True, dropout=True, dilation=True)
    actv4, pl4 = createConvolutionBlock(pl3, 128, pool=True, dropout=False, dilation=True)
    actv5, pl5 = createConvolutionBlock(pl4, 256, pool=True, dropout=True, dilation=True)
    actv6, pl6 = createConvolutionBlock(pl5, 512, pool=True, dropout=False, dilation=True)


    bridge = createConvolutionBlock(pl6, 1024, pool=False, dropout=False, dilation=False)


    dec1 = createDecoderBlock(bridge, actv6, 512, dropout=False, dilation=False)
    dec2 = createDecoderBlock(dec1, actv5, 256, dropout=True, dilation=False)
    dec3 = createDecoderBlock(dec2, actv4, 128, dropout=False, dilation=False)
    dec4 = createDecoderBlock(dec3, actv3, 64, dropout=True, dilation=False)
    dec5 = createDecoderBlock(dec4, actv2, 32, dropout=False, dilation=False)

    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(dec5)

    return Model(inputs, output)

def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning as a dictionary"""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # We have to convert the new value (0)


    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0"""
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize the images"""
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def data_augmentation(input_image, input_mask):
    input_image = tf.image.random_brightness(input_image, max_delta=0.1)
    input_image = tf.image.random_contrast(input_image, lower=0.1, upper=0.2)

    return input_image, input_mask

def load_dataset(dataset_path,training_data):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1024
    
    train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg", seed=42)
    train_dataset = train_dataset.map(parse_image)
    
    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg", seed=42)
    val_dataset =val_dataset.map(parse_image)

    dataset = {"train": train_dataset, "val": val_dataset}
    
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=42)
    dataset['train'] = dataset['train'].map(data_augmentation, num_parallel_calls=AUTOTUNE)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
  
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    return dataset




dataset_path = "ADEChallengeData2016/images/"
training_data = "training/"
val_data = "validation/"


trainset_size = len(glob(dataset_path + training_data + "*.jpg"))
valset_size = len(glob(dataset_path + val_data + "*.jpg"))


input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

dataset = load_dataset(dataset_path,training_data)


epochs = 20
train_steps = trainset_size // batch_size
validation_steps = valset_size // batch_size

model = build_model_architecture(input_size, N_CLASSES)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])


#model.summary()

#$ tensorboard --log_dir=logs/

callbacks = [
            ModelCheckpoint("pretreinedmodels/model.val_loss={val_loss:.5f}.h5", monitor='val_loss', verbose=1, save_best_model=True),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1),
            TensorBoard(log_dir="logs/{}".format(time()))
        ]

model.fit(dataset['train'],
            steps_per_epoch=train_steps,
            validation_data=dataset['val'],
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )
