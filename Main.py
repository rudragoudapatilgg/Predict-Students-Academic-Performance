import os
from numpy import matlib
from embedding4bert import Embedding4BERT
import re
import string
from CHCO import CHCO
from FHO import FHO
from Model_CNN import Model_CNN
from Model_DNN import Model_DNN
from Model_GCNN import Model_GCNN
from Model_LSTM import Model_Res_LSTM
from Objfun import *
from PlotResults_DS import *
from Plot_Results import *
from Proposed import Proposed
from SOA import SOA
from TSA import TSA

No_of_Dataset = 2


def extract_lexical_features(text):
    # Normalize the text
    text = text.lower()
    # Tokenize text by words
    words = re.findall(r'\b\w+\b', text)
    # Calculate word count
    word_count = len(words)
    # Calculate sentence count (using .!? as sentence delimiters)
    sentence_count = len(re.findall(r'[.!?]', text))
    # Average word length
    word_lengths = [len(word) for word in words]
    avg_word_length = int(sum(word_lengths) / word_count if word_count > 0 else 0)
    # Punctuation count
    punctuation_count = len([char for char in text if char in string.punctuation])
    # Unique word count
    unique_words = set(words)
    unique_word_count = len(unique_words)
    # Average sentence length (in words)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    feat = [word_count, sentence_count, avg_word_length, punctuation_count, unique_word_count, avg_sentence_length]

    return feat


# Read Dataset
an = 0
if an == 1:
    Df = pd.read_csv('english_dataset_wlabels.csv')
    data = Df.get(['answers']).values
    np.save('Data.npy', data)

# Feature Extraction using BERT
an = 0
if an == 1:
    prep = np.load('Data.npy', allow_pickle=True)  # Load the Preprocessing Data
    BERT = []
    for i in range(prep.shape[0]):
        print(i)
        emb4bert = Embedding4BERT("bert-base-cased")  # bert-base-uncased
        tokens, embeddings = emb4bert.extract_word_embeddings(prep[i][0])
        BERT.append(embeddings)
    np.save('BERT.npy', BERT)  # Save the BERT data

# Get Score Prediction
an = 0
if an == 1:
    data = np.load('BERT.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    feat = Model_Res_LSTM(data, Target)
    np.save('Lstm_Score.npy', feat)

# Feature Extraction using lexical and shallow
an = 0
if an == 1:
    text = np.load('Data.npy', allow_pickle=True)
    Feat = []
    for n in range(len(text)):
        print(n)
        texts = text[n][0]
        features = extract_lexical_features(texts)
        Feat.append(features)
    np.save('Lexical_Feat.npy', np.asarray(Feat))

# Optimization for Clustering
an = 0
if an == 1:
    Data = np.load('Lexical_Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Fuzziness parameter, Epsilon, Iteration
    xmin = matlib.repmat([2, 1, 10], Npop, 1)
    xmax = matlib.repmat([20, 10, 100], Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("FHO...")
    [bestfit1, fitness1, bestsol1, time] = FHO(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit2, fitness2, bestsol2, time1] = SOA(initsol, fname, xmin, xmax, Max_iter)

    print("TSA...")
    [bestfit3, fitness3, bestsol3, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

    print("CHCO...")
    [bestfit4, fitness4, bestsol4, time3] = CHCO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol3, bestsol4, bestsol5]
    np.save('BestSol_FCM.npy', BestSol)

# FCM Clustering
an = 0
if an == 1:
    data = np.load('Lexical_Feat.npy', allow_pickle=True)
    bests = np.load('BestSol_FCM.npy', allow_pickle=True)
    sol = np.round(bests[4, :]).astype(np.int16)
    # Define the number of clusters
    n_clusters = 5
    # Apply fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, n_clusters, int(sol[0]), error=0.005, maxiter=int(sol[2]), init=None
    )
    # Predict cluster membership for each data point
    cluster_membership = np.argmax(u, axis=0)
    np.save('FCM.npy', cluster_membership)

# Form Structured Data
an = 0
if an == 1:
    Lstm_Data = np.load('Lstm_Score.npy', allow_pickle=True)
    Lexical_Feat = np.load('Lexical_Feat.npy', allow_pickle=True)
    fcm_Data = np.load('FCM.npy', allow_pickle=True)
    Structured_Data = np.concatenate((Lstm_Data, Lexical_Feat, fcm_Data), axis=1)
    np.save('Formed_Data.npy', Structured_Data)


# Optimization for Text Generation
an = 0
if an == 1:
    Data = np.load('Formed_Data.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Stpes per Epochs, learning Rate
    xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 255, 0.99], Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("FHO...")
    [bestfit1, fitness1, bestsol1, time] = FHO(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit2, fitness2, bestsol2, time1] = SOA(initsol, fname, xmin, xmax, Max_iter)

    print("TSA...")
    [bestfit3, fitness3, bestsol3, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

    print("CHCO...")
    [bestfit4, fitness4, bestsol4, time3] = CHCO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', BestSol)

# Text Generation
an = 0
if an == 1:
    Feature = np.load('Formed_Data.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # loading step
    EVAL = []
    epoch = [100, 200, 300, 400, 500]
    for learn in range(len(epoch)):
        learnperc = round(Feature.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feature[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feature[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(learn, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :] = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn],
                                     sol)  # RLSTM With optimization
        Eval[5, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model CNN
        Eval[7, :], pred1 = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model DNN
        Eval[8, :], pred2 = Model_GCNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model GCNN
        Eval[9, :], pred3 = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target,
                                        epoch[learn])  # RLSTM Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Evaluate_Epochs.npy', EVAL)  # Save Eval all

# Read Dataset 2
an = 0
if an == 1:
    file_path = './Dataset_2'
    out_dir = os.listdir(file_path)
    Datas = []
    for i in range(len(out_dir)):
        file = file_path + '/' + out_dir[i]
        encoding = 'latin1'
        df = pd.read_csv(file, sep='\t', encoding=encoding, error_bad_lines=False)
        data = df.get(['essay']).values
        Datas.append(data)
    Concat_Data = np.concatenate((Datas[0], Datas[1], Datas[2]), axis=0)
    np.save('Data_2.npy', Concat_Data)

# Feature Extraction using BERT
an = 0
if an == 1:
    prep = np.load('Data_2.npy', allow_pickle=True)  # Load the Preprocessing Data
    BERT = []
    for i in range(prep.shape[0]):
        print(i)
        emb4bert = Embedding4BERT("bert-base-cased")  # bert-base-uncased
        tokens, embeddings = emb4bert.extract_word_embeddings(prep[i][0])
        BERT.append(embeddings)
    np.save('BERT_2.npy', BERT)  # Save the BERT data

# Get Score Prediction
an = 0
if an == 1:
    data = np.load('BERT_2.npy', allow_pickle=True)
    Target = np.load('Target_2.npy', allow_pickle=True)
    feat = Model_Res_LSTM(data, Target)
    np.save('Lstm_Score_2.npy', feat)

# Feature Extraction using lexical and shallow
an = 0
if an == 1:
    text = np.load('Data_2.npy', allow_pickle=True)
    Feat = []
    for n in range(len(text)):
        print(n)
        texts = text[n][0]
        if texts == str(texts):
            features = extract_lexical_features(texts)
            Feat.append(features)
    np.save('Lexical_Feat_2.npy', np.asarray(Feat))

# Optimization for Clustering
an = 0
if an == 1:
    Data = np.load('Lexical_Feat_2.npy', allow_pickle=True)
    Target = np.load('Target_2.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Fuzziness parameter, Epsilon, Iteration
    xmin = matlib.repmat([2, 1, 10], Npop, 1)
    xmax = matlib.repmat([20, 10, 100], Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("FHO...")
    [bestfit1, fitness1, bestsol1, time] = FHO(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit2, fitness2, bestsol2, time1] = SOA(initsol, fname, xmin, xmax, Max_iter)

    print("TSA...")
    [bestfit3, fitness3, bestsol3, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

    print("CHCO...")
    [bestfit4, fitness4, bestsol4, time3] = CHCO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol3, bestsol4, bestsol5]
    np.save('BestSol_FCM_2.npy', BestSol)

# FCM Clustering
an = 0
if an == 1:
    data = np.load('Lexical_Feat_2.npy', allow_pickle=True)
    bests = np.load('BestSol_FCM_2.npy', allow_pickle=True)
    sol = np.round(bests[4, :]).astype(np.int16)
    # Define the number of clusters
    n_clusters = 5
    # Apply fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T, n_clusters, int(sol[0]), error=0.005, maxiter=int(sol[2]), init=None
    )
    # Predict cluster membership for each data point
    cluster_membership = np.argmax(u, axis=0)
    np.save('FCM_2.npy', cluster_membership)

# Form Structured Data
an = 0
if an == 1:
    Lstm_Data = np.load('Lstm_Score_2.npy', allow_pickle=True)
    Lexical_Feat = np.load('Lexical_Feat_2.npy', allow_pickle=True)
    fcm_Data = np.load('FCM_2.npy', allow_pickle=True)
    Structured_Data = np.concatenate((Lstm_Data, Lexical_Feat, fcm_Data), axis=1)
    np.save('Formed_Data_2.npy', Structured_Data)

# Optimization for Text Generation
an = 0
if an == 1:
    Data = np.load('Formed_Data_2.npy', allow_pickle=True)
    Target = np.load('Target_2.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Stpes per Epochs, learning Rate
    xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 255, 0.99], Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("FHO...")
    [bestfit1, fitness1, bestsol1, time] = FHO(initsol, fname, xmin, xmax, Max_iter)

    print("SOA...")
    [bestfit2, fitness2, bestsol2, time1] = SOA(initsol, fname, xmin, xmax, Max_iter)

    print("TSA...")
    [bestfit3, fitness3, bestsol3, time2] = TSA(initsol, fname, xmin, xmax, Max_iter)

    print("CHCO...")
    [bestfit4, fitness4, bestsol4, time3] = CHCO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol3, bestsol4, bestsol5]
    np.save('BestSol_2.npy', BestSol)

# Text Generation
an = 0
if an == 1:
    Feature = np.load('Formed_Data_2.npy', allow_pickle=True)  # loading step
    Target = np.load('Target_2.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol_2.npy', allow_pickle=True)  # loading step
    EVAL = []
    epoch = [100, 200, 300, 400, 500]
    for learn in range(len(epoch)):
        learnperc = round(Feature.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feature[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feature[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(learn, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :] = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn],
                                     sol)  # RLSTM With optimization
        Eval[5, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model CNN
        Eval[7, :], pred1 = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model DNN
        Eval[8, :], pred2 = Model_GCNN(Train_Data, Train_Target, Test_Data, Test_Target, epoch[learn])  # Model GCNN
        Eval[9, :], pred3 = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target,
                                        epoch[learn])  # RLSTM Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Evaluates.npy', EVAL)  # Save Eval all

plotConvResults()
Plots_Results()
Plot_Results()
Table()
plot_ConvResults()
PlotsResults()
PlotResults()
Dataset_2_Table()
