import numpy as np
import yaml
import matplotlib.pyplot as plt
from ann import ANN
from fitness import evaluate_individual
from selection import selection
from mutation import mutate


population_size = 20
Generations = 100
mu = 5                              # number of best individuals kept unchanged (elites)
lambda_ = population_size - mu      # number of children made by crossover each generation
episodes_per_eval = 3               # arena layouts each robot is tested on per generation

with open("config.yaml") as f:
    config = yaml.safe_load(f)

rng = np.random.default_rng()
population = [ANN() for _ in range(population_size)]

History = []
Best_ANN = None
best_fitness_ever = -np.inf

plt.ion()

# loop over generations
for generation in range(Generations):
    # same seeds for every individual this generation, so the comparison is fair
    episode_seeds = rng.integers(0, 1_000_000, size=episodes_per_eval).tolist()

    fitness = np.zeros(len(population))
    for i, individual in enumerate(population):
        fitness[i], _ = evaluate_individual(individual, config, episode_seeds)

    # Store the best ANN seen so far
    best_idx = int(np.argmax(fitness))
    if fitness[best_idx] > best_fitness_ever:
        best_fitness_ever = fitness[best_idx]
        Best_ANN = ANN()
        Best_ANN.weights = population[best_idx].weights.copy()

    # for new generation: mu elites followed by lambda_ crossover children
    new_generation = selection(fitness, population, mu, lambda_)

    # Mutate only the children so the elites survive unchanged
    for child in new_generation[mu:]:
        child.weights = mutate(child.weights)

    print(f"Generation {generation}: avg = {fitness.mean():.3f}, Best Fitness = {fitness.max():.3f}")

    # append the information to the history
    History.append([generation, fitness.mean(), fitness.max()])

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

    # mutated children + elites become the new population for the next round
    population = new_generation

# Endloop

# saving the history in a CSV file
np.savetxt('fitness_history.csv', np.array(History), delimiter=",", header="gen,avg,best", comments="")

# saving the best ANN model
np.save('best_ann_model.npy', Best_ANN.weights)

plt.ioff()
plt.show()
