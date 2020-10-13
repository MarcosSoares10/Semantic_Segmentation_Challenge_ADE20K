from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import numpy as np

def createConvolutionBlock(inputs, filters, pool = True):
    convA = Conv2D(filters, 3, padding="same")(inputs)
    batchA = BatchNormalization()(convA)
    actvA = Activation("relu")(batchA)

    convB = Conv2D(filters, 3, padding="same")(actvA)
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


model = build_model_architecture((256, 256, 3), 10)
model.summary()
    
