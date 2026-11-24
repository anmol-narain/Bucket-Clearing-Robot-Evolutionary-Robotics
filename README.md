# Bucket-Clearing-Robot-Evolutionary-Robotics
This project evolves an artificial neural network (ANN) that controls a simple mobile robot. The robot pushes a random number of randomly placed buckets out of an arena through the gaps in its outer wall.

## Controller and genotype

- Encode the controller as a 13 × 2 matrix of 26 real-valued genes. The 13 rows account for 12 inputs and one bias term, and the two columns correspond to the two ANN outputs.
- Use the genotype directly as the ANN weight matrix. Start with a memoryless controller to keep the design simple and revisit it if necessary.
- The exact ordering of the weight and bias genes is still open.

## Fitness function

```text
fitness = 5 * (n / N) + s / 60 - w / 60 - max(5, o) / 55
```

| Variable | Meaning |
| --- | --- |
| `n` | Number of buckets pushed outside the arena |
| `N` | Total number of buckets in the arena |
| `s` | Seconds the robot spends moving a bucket |
| `w` | Seconds the robot spends touching a wall |
| `o` | Seconds the robot spends outside the arena |

Dividing by `N` normalizes the bucket-clearing reward across arenas with different bucket counts. The group identified idling or spinning in place, receiving credit for touching a bucket without moving it, and clearing one bucket before stopping as behaviors to check during evaluation. The written formula gives an idle robot with all counters at zero a fitness of `-5/55`, while the notes say zero. Confirm the intended outside-arena penalty before implementation.

## Evolutionary operators

- **Parent selection** uses rank-based selection.
- **Survivor selection** uses the `(μ + λ)` scheme. Select the best `μ` individuals from the combined pool of `μ` parents and `λ` offspring.
- **Crossover** uses arithmetic crossover. Compare runs with and without crossover to assess its effect.
- **Mutation** adds zero-mean Gaussian noise with standard deviation `σ/20`, where `σ` is the standard deviation of the zero-mean weight initialization. The small step size preserves useful weights while allowing cumulative changes across generations.

## Provisional run settings

| Setting | Provisional value |
| --- | ---: |
| Population size | 50 |
| Number of generations | 50 |

The evaluation episode count and duration, rank-selection parameters, offspring count, crossover rate, mutation rate, and initialization standard deviation still need values.
