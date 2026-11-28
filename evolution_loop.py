import numpy as np
import yaml
import matplotlib.pyplot as plt
from ann import ANN
from fitness import evaluate_individual
from selection import selection
from mutation import mutate
from concurrent.futures import ProcessPoolExecutor
import os

def _eval_worker(args):
    individual, cfg, seeds = args
    return evaluate_individual(individual, cfg, seeds)

EVOLUTION_SEED = 10  # Change this to select a different evolution run.
population_size = 20
Generations = 25
mu = 5                              # number of best individuals kept unchanged (elites)
lambda_ = population_size - mu      # number of children made by crossover each generation
episodes_per_eval = 2               # arena layouts each robot is tested on per generation

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ANN initialization, parent selection, and mutation use NumPy's global RNG.
np.random.seed(EVOLUTION_SEED)
rng = np.random.default_rng(EVOLUTION_SEED)
population = [ANN() for _ in range(population_size)]

History = []
Best_ANN = None
best_fitness_ever = -np.inf

plt.ion()

with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    for generation in range(Generations):
        # Same seeds for every individual this generation, so the comparison is fair
        episode_seeds = rng.integers(0, 1_000_000, size=episodes_per_eval).tolist()

        tasks = [(ind, config, episode_seeds) for ind in population]
        results = list(pool.map(_eval_worker, tasks))

        fitness = np.array([r[0] for r in results], dtype=float)
        buckets_cleared = np.array(
            [np.mean([ep["buckets_out"] for ep in r[1]]) for r in results],
            dtype=float,
        )

        # Store the best ANN seen so far
        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > best_fitness_ever:
            best_fitness_ever = fitness[best_idx]
            Best_ANN = ANN()
            Best_ANN.weights = population[best_idx].weights.copy()

        # New generation: mu elites followed by lambda_ crossover children
        new_generation = selection(fitness, population, mu, lambda_)

        # Mutate only the children so the elites survive unchanged
        for child in new_generation[mu:]:
            child.weights = mutate(child.weights, mutation_rate=0.2, std=0.13)

        print(
            f"Generation {generation}: avg fitness = {fitness.mean():.6g}, "
            f"best fitness = {fitness[best_idx]:.6g}, "
            f"buckets cleared per episode: population avg = {buckets_cleared.mean():.3f}, "
            f"top-scoring controller = {buckets_cleared[best_idx]:.3f}"
        )

        History.append([
            generation, fitness.mean(), fitness[best_idx],
            buckets_cleared.mean(), buckets_cleared[best_idx],
        ])

        population = new_generation

# Endloop
# Plotting
h = np.array(History)
plt.clf()
plt.plot(h[:, 0], h[:, 1], label='Average Fitness')
plt.plot(h[:, 0], h[:, 2], label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations')
plt.legend()
plt.pause(0.01)
# saving the history in a CSV file
np.savetxt(
    'fitness_history.csv', h, delimiter=",",
    header="gen,avg,best,avg_buckets,top_scoring_buckets", comments="",
)

# saving the best ANN model
np.save('best_ann_model.npy', Best_ANN.weights)

plt.ioff()
plt.show()
