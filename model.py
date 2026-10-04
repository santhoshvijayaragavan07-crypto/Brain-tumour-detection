import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lightweight_cnn(num_classes):
    inputs = layers.Input((224,224,3), name="mri_input")
    x = layers.Conv2D(24,3,strides=2,padding="same",use_bias=False,name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    for i,(filters,stride) in enumerate([(32,1),(64,2),(96,2),(128,2)],1):
        x = layers.DepthwiseConv2D(3,strides=stride,padding="same",
                                   use_bias=False,name=f"dw_{i}")(x)
        x = layers.BatchNormalization(name=f"dw_bn_{i}")(x)
        x = layers.ReLU(name=f"dw_relu_{i}")(x)
        x = layers.Conv2D(filters,1,padding="same",use_bias=False,name=f"pw_{i}")(x)
        x = layers.BatchNormalization(name=f"pw_bn_{i}")(x)
        x = layers.ReLU(name=f"pw_relu_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.25,name="dropout")(x)
    outputs = layers.Dense(num_classes,activation="softmax",name="classifier")(x)
    return Model(inputs,outputs,name="LightweightBrainTumorCNN")
