import time
import numpy as np


def levy_flight(current_position, dim):
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
            np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(loc=0, scale=sigma, size=dim)
    v = np.random.normal(loc=0, scale=1, size=dim)
    step = u / np.power(np.abs(v), 1 / beta)
    next_position = current_position + 0.01 * step
    return next_position


# Fire Hawk Optimizer (FHO)
def FHO(population, func, lb, ub, num_iter):
    pop_size, dim = population.shape
    alpha = 0.5
    beta = 0.5
    gamma = 0.8
    delta = 0.2
    fitness = func(population[:])
    fbest = np.random.rand()
    Convergence_curve = np.zeros(num_iter)
    bestsol = float('inf')
    best_fit = np.zeros((dim, 1))

    ct = time.time()

    for t in range(num_iter):
        for i in range(pop_size):
            current_position = population[i]
            next_position = levy_flight(current_position, dim)
            if np.any(next_position > 1) or np.any(next_position < -1):
                next_position = np.clip(next_position, -1, 1)
            fitness_current = func(current_position)
            fitness_next = func(next_position)
            if fitness_next < fitness_current:
                population[i] = next_position
                fitness[i] = fitness_next
            else:
                p = np.exp(-beta * (fitness_next - fitness_current) / np.abs(current_position.mean()))
                if np.random.uniform() < p:
                    population[i] = next_position
                    fitness[i] = fitness_next
        best_position = population[np.argmin(fitness)]
        for i in range(pop_size):
            r = np.random.uniform()
            if r < alpha:
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                if best_position[0] < fbest:
                    fbest = best_position[0]
                if r1 < 0.5:
                    population[i] = best_position + gamma * (
                            population[i] - population[np.random.randint(pop_size)]) + delta * (
                                            np.random.uniform(size=dim) - 0.5)
                else:
                    population[i] = population[i] + beta * np.abs(r2 * best_position - population[i]) + gamma * (
                            population[np.random.randint(pop_size)] - population[i]) + delta * (
                                            np.random.uniform(size=dim) - 0.5)

        Convergence_curve[t] = best_fit[0]
    best_fit = Convergence_curve[Convergence_curve.shape[0] - 1]
    ct = time.time() - ct
    return best_fit, Convergence_curve, bestsol, ct
