from keras.layers import Input, Dense, BatchNormalization, Activation, Conv1D, Dense, Dropout, Dense, Dropout
import numpy as np
from keras.models import Model
from Evaluation import evaluation


def GCN_block(inputs, feature_dims, activation):
    # Graph Convolution Layer
    outputs = Conv1D(feature_dims, 1, activation=None, use_bias=False)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation)(outputs)
    return outputs


def Model_GCNN(train_data, Train_Target, test_data, Test_Target, epoch):

    Train_x = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    Test_x = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

    # Define the input shape
    inputs = Input(shape=(1, Train_x.shape[2]))
    # Define the model architecture
    x = GCN_block(inputs, 64, 'relu')

    x = Dropout(0.5)(x)
    x = GCN_block(x, 64, 'relu')
    x = Dropout(0.5)(x)
    outputs = GCN_block(x, Train_Target.shape[1], 'softmax')

    # Define the model and compile it
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(Train_x, Train_Target, epochs=epoch, batch_size=32)

    pred = model.predict(Test_x)
    Eval = evaluation(pred, Test_Target)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return Eval, pred

