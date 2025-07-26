import numpy as np
from Evaluation import evaluation
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


def Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Epoch):
    padded_sequences = []
    for m in range(Train_Data.shape[0]):
        Data = Train_Data[m]
        texts = Data

        # Tokenize the text data
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Pad sequences to make them the same length
        max_len = max([len(seq) for seq in sequences])
        Datas = pad_sequences(sequences, maxlen=max_len)
        re_data = np.reshape(Datas, (Datas.shape[0] * Datas.shape[1]))
        min_value = 20
        padded_sequences.append(re_data[:min_value])
    Test_padded_sequences= []
    for n in range(Test_Data.shape[0]):
        Data = Test_Data[n]
        texts = Data

        # Tokenize the text data
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # Pad sequences to make them the same length
        max_len = max([len(seq) for seq in sequences])
        Datas = pad_sequences(sequences, maxlen=max_len)
        re_data = np.reshape(Datas, (Datas.shape[0] * Datas.shape[1]))
        min_value = 20
        Test_padded_sequences.append(re_data[:min_value])

    # Build the CNN model
    model = Sequential([
        Embedding(1000, 128, input_length=20),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(Train_Target.shape[1], activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # Train the model
    model.fit(np.asarray(padded_sequences), Train_Target, epochs=10)
    pred = model.predict(np.asarray(Test_padded_sequences))
    Eval = evaluation(pred, Test_Target)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return Eval, pred

