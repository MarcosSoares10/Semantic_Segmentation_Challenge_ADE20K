#Dataset
#http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate,concatenate,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from glob import glob
import datetime

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 151
batch_size = 60

def createConvolutionBlock(inputs, filters, pool = True):
    initializer = 'he_normal'
    convA = Conv2D(filters, 3, padding="same",kernel_initializer=initializer)(inputs)
    batchA = BatchNormalization()(convA)
    actvA = Activation("relu")(batchA)

    convB = Conv2D(filters, 3, padding="same",kernel_initializer=initializer)(actvA)
    batchB = BatchNormalization()(convB)
    actvB = Activation("relu")(batchB)

    if pool:
        pl = MaxPool2D((2,2))(actvB)
        return actvB, pl
    else:
        return actvB

def createDecoderBlock(inputlayer, concatlayer, filters):
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(inputlayer) # "nearest", "bilinear", "bicubic"
    c1 = Concatenate()([u1, concatlayer])
    conv_block = createConvolutionBlock(c1, filters, pool=False)

    return conv_block

def build_model_architecture(shape, num_classes):
    inputs = Input(shape)

    actv1, pl1 = createConvolutionBlock(inputs, 16, pool=True)
    actv2, pl2 = createConvolutionBlock(pl1, 32, pool=True)
    actv3, pl3 = createConvolutionBlock(pl2, 48, pool=True)
    actv4, pl4 = createConvolutionBlock(pl3, 64, pool=True)

    bridge = createConvolutionBlock(pl4, 128, pool=False)

    dec1 = createDecoderBlock(bridge, actv4, 64)
    dec2 = createDecoderBlock(dec1, actv3, 48)
    dec3 = createDecoderBlock(dec2, actv2, 32)
    dec4 = createDecoderBlock(dec3, actv1, 16)

    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(dec4)

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

def load_dataset(dataset_path,training_data):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1000
    train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg", seed=42)
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg", seed=42)
    val_dataset =val_dataset.map(parse_image)

    dataset = {"train": train_dataset, "val": val_dataset}
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=42)
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

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss = loss, metrics=['accuracy'])

callbacks = [
            ModelCheckpoint("pretreinedmodels/model.val_loss={val_loss:.5f}.h5", monitor='val_loss', verbose=1, save_best_model=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        ]

model.fit(dataset['train'],
            steps_per_epoch=train_steps,
            validation_data=dataset['val'],
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )
 
