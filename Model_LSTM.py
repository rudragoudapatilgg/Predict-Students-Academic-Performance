import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Add
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K


# Residual LSTM block
def resblock(inputs, filters, strides):
    y = inputs  # Shortcut path

    # Main path
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Fit shortcut path dimenstions
    if strides > 1:
        y = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=3,
            strides=strides,
            padding='same',
        )(y)
        y = tf.keras.layers.BatchNormalization()(y)

    # Concatenate paths
    x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.ReLU()(x)

    return x


def build_res_lstm_model(input_shape, num_classes, units=64, num_blocks=3):
    inputs = Input(shape=input_shape)

    x = inputs
    for _ in range(num_blocks):
        x = resblock(x, 64, 1)

    x = LSTM(units)(x)  # Final LSTM layer without return_sequences
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Model_Res_LSTM(Data, Target):
    Train_x = np.reshape(Data, (Data.shape[0], Data.shape[1], 1))
    model = build_res_lstm_model((1, Train_x.shape[2]), 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_x, Target, epochs=10)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    test = Data[:][np.newaxis, ...]
    tets = test[:, :, :]
    layer_out = np.asarray(functors[4]([tets])).squeeze()
    return layer_out
