import numpy as np
import matplotlib.pyplot as plt
from ann import ANN
from fitness import get_fitness
from crossover import crossover
from selection import selection
from mutation import mutate


population_size = 20 
Generations = 100

population =  [ANN() for _ in range(population_size )]

History = []

# loop over generations
for generation in range(Generations):
    #Fitness = np.array 
    fitness = np.zeros(len(population))

    for i, individual in enumerate(ANN):
        fitness[i] = get_fitness(individual)

    

    # Store the best ANN for this generation
    Best_ANN = ANN[np.argmax(fitness)]

    # for new generation
    new_generation = crossover(selection(population, fitness))

    # Mutate the new generation
    new_generation = mutate(new_generation)

    print(f"Generation {generation}:avg = {fitness.mean():.3f}, Best Fitness = {fitness.max():.3f}")

    # append the information to the history
    History.append([generation, fitness.mean(), fitness.max()])

    #PLoting
    h = np.array(History)
    plt.clf()
    plt.plot(h[:, 0], h[:, 1], label='Average Fitness')
    plt.plot(h[:, 0], h[:, 2], label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.pause(0.01)

    # muteted children becomes the new population for the next round
    ANN = new_generation

# Endloop

# saving the histopry ion CSV file
np.savetxt('fitness_history.csv', np.array(History), delimiter=",",header = "gen,avg,best", comments="")

# saving the best ANN model 
np.save('best_ann_model.npy', Best_ANN.weights)

plt.show()

