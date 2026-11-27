# Bucket-Clearing-Robot-Evolutionary-Robotics
This project evolves an artificial neural network (ANN) that controls a simple mobile robot. The robot pushes a random number of randomly placed buckets out of an arena through the gaps in its outer wall.

## Controller and genotype

- Encode the controller as a 13 × 2 matrix of 26 real-valued genes. The 13 rows account for 12 inputs and one bias term, and the two columns correspond to the two ANN outputs.
- Use the genotype directly as the ANN weight matrix. Start with a memoryless controller to keep the design simple and revisit it if necessary.
- The exact ordering of the weight and bias genes is still open.

## Fitness function

```text
fitness = 5 * (n / N) + 0.5 * (s / T) + 0.5 * a - w / T - i / T - outside_penalty
```

| Variable | Meaning |
| --- | --- |
| `n` | Number of buckets pushed outside the arena |
| `N` | Total number of buckets in the arena |
| `s` | Seconds the robot spends moving a bucket |
| `a` | Fraction of the initial gap to the nearest bucket that the robot closes at its closest approach, capped at 1 |
| `w` | Seconds the robot spends touching a wall |
| `i` | Seconds when both the robot's linear speed and the speed at its rim from turning are below 0.02 m/s |
| `o` | Seconds the robot spends outside the arena |
| `T` | Episode time limit (60 seconds by default) |

Dividing by `N` normalizes the bucket-clearing reward across arenas with different bucket counts. The outside penalty is zero if the robot stays inside and `2 + 3 * (o / T)` if it exits. A full episode spent idle incurs a 1-point penalty. The group identified spinning in place, receiving credit for touching a bucket without moving it, and clearing one bucket before stopping as behaviors to check during evaluation.

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
