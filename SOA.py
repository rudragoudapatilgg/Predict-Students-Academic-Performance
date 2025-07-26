import time

import numpy as np


# Skill Optimization Algorithm (SOA)
def SOA(SearchAgents, fobj, VRmin, VRmax, Max_iterations):
    N, dimension = SearchAgents.shape[0], SearchAgents.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    fitness = fobj(SearchAgents[:])
    X = SearchAgents
    fit = fitness

    # Initialize variables to keep track of the best solution
    fbest = np.inf
    Xbest = np.zeros(dimension)
    SOA_curve = np.zeros(Max_iterations)

    ct = time.time()
    for t in range(Max_iterations):
        # Update the best member and worst member
        best = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            Xbest = X[blocation, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            Xbest = X[blocation, :]

        # Update SOA population
        for i in range(SearchAgents.shape[0]):
            # Phase 1: Exploration
            K = np.where(fit < fit[i])[0]
            if K.size != 0:
                KK = np.random.choice(K, size=1)
            else:
                KK = i

            expert = X[KK, :]
            if np.random.rand() < 0.5:
                I = round(np.random.rand())
                RAND = np.random.rand()
            else:
                I = np.round(np.random.rand(dimension))
                RAND = np.random.rand(dimension)

            X_P1 = X[i, :] + RAND * (expert - I * X[i, :])  # Eq. (3)
            X_P1 = np.clip(X_P1, lb, ub)

            # Update position based on Eq (4)
            F_P1 = fitness[:X_P1.shape[1]]
            if F_P1[0] <= fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1[0]

        # Phase 2: Exploitation (local search)
        for i in range(SearchAgents.shape[0]):
            if np.random.rand() < 0.5:
                X_P2 = X[i, :] + ((1 - 2 * np.random.rand(dimension)) / (t + 1)) * X[i, :]  # Eq. (5)
                X_P2 = np.clip(X_P2, lb, ub)
            else:
                X_P2 = X[i, :] + (lb / (t + 1)) + np.random.rand() * (
                        ub / (t + 1) - lb / (t + 1))  # Eq. (5)
                X_P2 = np.clip(X_P2, lb / (t + 1), ub / (t + 1))
                X_P2 = np.clip(X_P2, lb, ub)

            # Update position based on Eq (6)
            F_P2 = fitness[:X_P2.shape[0]]
            if F_P2[0] < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2[0]

        # Record the best score at each iteration
        SOA_curve[t] = fbest

    Best_score = fbest
    Best_pos = Xbest
    ct = time.time() - ct

    return Best_score, SOA_curve, Best_pos, ct
