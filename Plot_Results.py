import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'FHO', 'SOA-ARes-LSTM', 'EOO-ARes-LSTM', 'CHC-ARes-LSTM', 'SRVCC-ARes-LSTM']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[:, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure(facecolor='#9ecae1')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.yaxis.grid()
    ax.set_facecolor("#deebf7")
    length = np.arange(50)
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='FHO-ARes-LSTM')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='SOA-ARes-LSTM')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='EOO-ARes-LSTM')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='CHC-ARes-LSTM')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='SRVCC-ARes-LSTM')
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def Table():
    eval = np.load('Eval_LearningRate.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DNN', 'RNN', 'LSTM', 'Bi-LSTM', 'SRVCC-ARes-LSTM']
    Classifier = ['TERMS', 'MPRL', 'DNN', 'GCN', 'Res_LSTM', 'SRVCC-ARes-LSTM']
    Terms = ['Accuracy', 'TS', 'MK', 'FOR', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR',
             'F1 Sscore', 'MCC', 'PT',
             'BA', 'FM', 'BM', 'PLHR', 'lrminus', 'DOR', 'Prevalence']
    Learning_Rate = [0.01, 0.11, 0.21, 0.31, 0.41]
    for i in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            value = eval[i, k, :, 4:]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[0:5])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, 0:5])
            print('-------------------------------------------------- Dataset', i + 1, 'Learning_Rate',
                  Learning_Rate[k],
                  'Classification Comparison',
                  '--------------------------------------------------')
            print(Table)


def Plots_Results():
    eval = np.load('Evaluate_Epochs.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC',
             'FOR', 'PT',
             'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0, 1, 2, 3, 8, 9, 10, 13, 15]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            epoch = [100, 200, 300, 400]
            fig = plt.figure(facecolor='#bdbdbd')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            ax.xaxis.grid()
            ax.yaxis.grid()
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Epoch')
            plt.plot(epoch, Graph[:, 0], color='#00FFFF', linewidth=8, marker='*',
                     markerfacecolor='#B22222', markersize=20,
                     label="FHO-ARes-LSTM")
            plt.plot(epoch, Graph[:, 1], color='#7FFF00', linewidth=8, marker='*',
                     markerfacecolor='#FF4500', markersize=20,
                     label="SOA-ARes-LSTM")
            plt.plot(epoch, Graph[:, 2], color='#808000', linewidth=8, marker='*', markerfacecolor='#8B008B',
                     markersize=20,
                     label="EOO-ARes-LSTM")
            plt.plot(epoch, Graph[:, 3], color='#CD1076', linewidth=8, marker='*', markerfacecolor='b',
                     markersize=20,
                     label="CHC-ARes-LSTM")
            plt.plot(epoch, Graph[:, 4], color='k', linewidth=8, marker='o', markerfacecolor='black',
                     markersize=15,
                     label="SRVCC-ARes-LSTM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True, facecolor='#f0f0f0')

            plt.xticks(epoch, ('100', '200', '300', '400'), fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel('Epoch', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(facecolor='#bdbdbd')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            ax.yaxis.grid()
            ax.set_facecolor("#f0f0f0")
            fig.canvas.manager.set_window_title('Epochs')
            X = np.arange(4)
            ax.bar(X + 0.00, Graph[:, 5], color='#FFC125', edgecolor='w', width=0.15, label="MPRL")
            ax.bar(X + 0.15, Graph[:, 6], color='#8DEEEE', edgecolor='w', width=0.15, label="DNN")
            ax.bar(X + 0.30, Graph[:, 7], color='#FF1493', edgecolor='w', width=0.15, label="GCN")
            ax.bar(X + 0.45, Graph[:, 8], color='lime', edgecolor='w', width=0.15, label="Res-LSTM")
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="SRVCC-ARes-LSTM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True, facecolor='#f0f0f0')
            plt.xticks(X + 0.25, ('100', '200', '300', '400'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='k')
            plt.xlabel('Epoch', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Results():
    Eval = np.load('Eval_Summ.npy', allow_pickle=True)
    Rouge = ['Rouge-1', 'Rouge-2', 'Rouge-l']
    Terms = ['F1Score', 'Precision', 'Recall']
    Algorithms = ['FHO-ARes-LSTM', 'SOA-ARes-LSTM', 'EOO-ARes-LSTM', 'CHC-ARes-LSTM', 'SRVCC-ARes-LSTM']
    Methods = ['MPRL', 'DNN', 'GCN', 'Res-LSTM', 'SRVCC-ARes-LSTM']
    Value = Eval[:, 0, :]

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Terms)
    for j in range(len(Algorithms)):
        Table.add_column(Algorithms[j], Value[j, :])
    print('-------------------------------------------------- Dataset-' + str(
        0 + 1) + ' Algorithm Comparison(Rouge-1) ',
          ' --------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Terms)
    for j in range(len(Methods)):
        Table.add_column(Methods[j], Value[j + 5, :])
    print('-------------------------------------------------- Dataset-' + str(
        0 + 1) + ' Methods Comparison(Rouge-1) ',
          ' --------------------------------------------------')
    print(Table)

    for k in range(len(Rouge)):
        fig = plt.figure(facecolor='#a1d99b')
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.yaxis.grid()
        ax.set_facecolor("#e5f5e0")
        X = np.arange(3)
        ax.bar(X + 0.00, Eval[0, k, :], color='#00FFFF', edgecolor='w', width=0.15, label="FHO-ARes-LSTM")
        ax.bar(X + 0.15, Eval[1, k, :], color='#FF00FF', edgecolor='w', width=0.15, label="SOA-ARes-LSTM")
        ax.bar(X + 0.30, Eval[2, k, :], color='#aa23ff', edgecolor='w', width=0.15, label="EOO-ARes-LSTM")
        ax.bar(X + 0.45, Eval[3, k, :], color='#7FFF00', edgecolor='w', width=0.15, label="CHC-ARes-LSTM")
        ax.bar(X + 0.60, Eval[4, k, :], color='#4B0082', edgecolor='w', width=0.15, label="SRVCC-ARes-LSTM")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                   ncol=3, fancybox=True, shadow=True, facecolor='#e5f5e0')
        plt.xticks(X + 0.30, ('F1 Score', 'Precision', 'Recall'), fontname="Arial", fontsize=12, fontweight='bold',
                   color='k')
        plt.ylabel(Rouge[k], fontname="Arial", fontsize=12, fontweight='bold', color='k')
        path1 = "./Results/Dataset_%s_%s_Algorithm.png" % (str(0 + 1), Rouge[k])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure(facecolor='#a1d99b')
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        ax.yaxis.grid()
        ax.set_facecolor("#e5f5e0")
        X = np.arange(3)
        ax.bar(X + 0.00, Eval[5, k, :], color='#02ccfe', edgecolor='w', width=0.15, label="MPRL")
        ax.bar(X + 0.15, Eval[6, k, :], color='#aa23ff', edgecolor='w', width=0.15, label="DNN")
        ax.bar(X + 0.30, Eval[7, k, :], color='#21fc0d', edgecolor='w', width=0.15, label="GCN")
        ax.bar(X + 0.45, Eval[8, k, :], color='#fcb001', edgecolor='w', width=0.15, label="Res-LSTM")
        ax.bar(X + 0.60, Eval[9, k, :], color='#4B0082', edgecolor='w', width=0.15, label="SRVCC-ARes-LSTM")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                   ncol=3, fancybox=True, shadow=True, facecolor='#e5f5e0')
        plt.xticks(X + 0.30, ('F1 Score', 'Precision', 'Recall'), fontname="Arial", fontsize=12, fontweight='bold',
                   color='k')
        plt.ylabel(Rouge[k], fontname="Arial", fontsize=12, fontweight='bold', color='k')
        path1 = "./Results/Dataset_%s_%s_Method.png" % (str(0 + 1), Rouge[k])
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plots_Results()
    Plot_Results()
    Table()
