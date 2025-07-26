import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from Evaluation import evaluation


def Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch):
    # Build the DNN model
    model = Sequential()
    model.add(Dense(64, input_dim=25, activation='relu'))  # Input layer + hidden layer
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(Train_Target.shape[1], activation='softmax'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    TrainX = np.asarray(Train_Data).astype(np.float32)
    TestX = np.asarray(Test_Data).astype(np.float32)
    model.fit(TrainX, Train_Target, epochs=epoch)
    pred = model.predict(TestX)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_Target)

    return Eval, pred


