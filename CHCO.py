import time

import numpy as np


def levy_flight(beta, D):
    """Generate a Levy flight step."""
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, D)
    v = np.random.normal(0, 1, D)
    return u / abs(v) ** (1 / beta)


def update_positions_cock(pos, dance, r1, best_pos, use_best):
    """Update positions of cocks based on their role."""
    if use_best:
        return pos + dance * r1
    else:
        return best_pos * levy_flight(1.5, len(pos))


def update_positions_hen(pos, r2, a, xb, xc, use_best):
    """Update positions of hens based on proximity to best or random cock."""
    if r2 < 0.5:
        return pos + a * r2 * (xb - pos)
    else:
        return xc + a * r2 * (xc - pos)


def update_positions_chicken(pos, xh, xb, xw, r3, r4, a, p1, lb, ub):
    """Update positions of chickens with different strategies."""
    if r3 < p1:
        return xh + a * np.random.uniform(0, 1) * (xh - pos)
    elif r4 < 0.5:
        return np.random.uniform(lb, ub, len(pos))
    else:
        return pos + a * np.random.uniform(0, 1) * (xw - pos)


#  Cock-hen-chicken Optimizer (CHC)
def CHCO(positions, fobj, VRmin, VRmax, maxIter):
    N, m = positions.shape[0], positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = fobj(positions[:])
    indices = np.argsort(fitness)
    cocks, hens, chickens = indices[:m], indices[m:m + N], indices[m + N:]
    best_pos = np.zeros((m, 1))
    best_cock_fitness = float('inf')
    Convergence_curve = np.zeros((maxIter, 1))

    ct = time.time()
    for t in range(maxIter):
        # Exploration and exploitation
        for idx in cocks:
            r1 = np.random.uniform(0, 1)
            positions[idx] = update_positions_cock(positions[idx], 0.8, r1, best_pos, idx == cocks[0])

        for idx in hens:
            r2, r3 = np.random.uniform(0, 1, 2)
            xc = positions[np.random.choice(cocks)]
            positions[idx] = update_positions_hen(positions[idx], r2, 2 * (1 - maxIter), best_pos, xc, r3 < 0.5)

        for idx in chickens:
            r3, r4 = np.random.uniform(0, 1, 2)
            xh = positions[np.random.choice(hens)]
            xw = positions[np.argmax(fitness)]
            positions[idx] = update_positions_chicken(positions[idx], xh, best_pos, xw, r3, r4, 2 * (1 - maxIter),
                                                      0.1, lb, ub)

        # Update fitness
        fitness = np.array([fobj(pos) for pos in positions])
        indices = np.argsort(fitness)
        cocks, hens, chickens = indices[:m], indices[m:m + N], indices[m + N:]
        best_pos = positions[cocks[0]]

        Convergence_curve[t] = best_cock_fitness
        t = t + 1
    best_cock_fitness = Convergence_curve[maxIter - 1][0]
    ct = time.time() - ct

    return best_cock_fitness, Convergence_curve, best_pos, ct
