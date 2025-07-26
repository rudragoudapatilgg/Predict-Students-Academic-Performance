import time

import numpy as np


def TSA(Search_Agents, objective, VRmin, VRmax, Max_iterations):
    N, dimensions = Search_Agents.shape[0], Search_Agents.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    Position = np.zeros((dimensions, 1))
    Score = np.inf
    Positions = Search_Agents
    Convergence = np.zeros(Max_iterations)
    t = 0
    ct = time.time()
    while t < Max_iterations:
        for i in range(Positions.shape[0]):
            # Check for bounds and correct if necessary
            Flag4Upperbound = Positions[i, :] > ub
            Flag4Lowerbound = Positions[i, :] < lb

            Positions[i, :] = (Positions[i, :] * ~(Flag4Upperbound | Flag4Lowerbound) +
                               ub * Flag4Upperbound +
                               lb * Flag4Lowerbound)

            fitness = objective(Positions[i, :])

            if fitness < Score:
                Score = fitness
                Position = Positions[i, :]

        # Randomly select xr within a range
        xmin = 1
        xmax = 4
        xr = int(xmin + np.random.rand() * (xmax - xmin))

        # Update positions based on the TSA algorithm
        for i in range(Positions.shape[0]):
            for j in range(Positions.shape[1]):
                A1 = (np.random.rand() + np.random.rand() - (2 * np.random.rand())) / xr
                c2 = np.random.rand()

                if i == 0:
                    c3 = np.random.rand()
                    if c3 >= 0:
                        d_pos = abs(Position[j] - c2 * Positions[i, j])
                        Positions[i, j] = Position[j] + A1 * d_pos
                    else:
                        d_pos = abs(Position[j] - c2 * Positions[i, j])
                        Positions[i, j] = Position[j] - A1 * d_pos
                else:
                    c3 = np.random.rand()
                    if c3 >= 0:
                        d_pos = abs(Position[j] - c2 * Positions[i, j])
                        Pos = Position[j] + A1 * d_pos
                    else:
                        Pos = Position[j] - A1 * d_pos

                    Positions[i, j] = (Pos + Positions[i - 1, j]) / 2

        t += 1
        Convergence[t - 1] = Score  # Store the convergence score for the current iteration
    ct = time.time() - ct

    return Score, Convergence, Position, ct
