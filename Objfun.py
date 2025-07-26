import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
import skfuzzy as fuzz
from Model_RLSTM import Model_RLSTM


def objfun(Soln):
    Data = Global_Vars.Data
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            n_clusters = 5
            # Apply fuzzy c-means clustering
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                Data.T, n_clusters, int(sol[0]), error=0.005, maxiter=int(sol[2]), init=None
            )
            cluster_membership = np.argmax(u, axis=0)
            accuracy = (Target - cluster_membership) / Target
            Fitn[i] = 1 / accuracy
        return Fitn
    else:
        sol = Soln
        n_clusters = 5
        # Apply fuzzy c-means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            Data.T, n_clusters, int(sol[0]), error=0.005, maxiter=int(sol[2]), init=None
        )
        # Predict cluster membership for each data point
        cluster_membership = np.argmax(u, axis=0)
        accuracy = (Target - cluster_membership) / Target
        Fitn = 1 / accuracy
        return Fitn


def objfun_cls(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, Pred = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(Pred, Test_Target)
            Fitn[i] = (1 / Eval[4]) + Eval[9]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, Pred = Model_RLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(Pred, Test_Target)
        Fitn = (1 / Eval[4]) + Eval[9]
        return Fitn
