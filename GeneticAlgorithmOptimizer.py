import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")


class GeneticAlgorithmOptimizer:
    def __init__(self,
                 population_size=200,
                 elite_size=20,
                 mutation_rate=0.02,
                 tournament_size=5,
                 crossover_method='order',
                 mutation_method='swap'):
        """
        Genetic Algorithm optimizer for the Traveling Salesman Problem.
        Finds the minimum (or maximum) cost route visiting all nodes exactly once.

        :param population_size:   number of individuals (routes) in each generation
        :param elite_size:        number of top individuals carried to the next generation unchanged
        :param mutation_rate:     probability [0-1] of mutating each individual
        :param tournament_size:   candidates compared in each tournament-selection round
        :param crossover_method:  'order' (OX) or 'pmx' (Partially Mapped)
        :param mutation_method:   'swap' | 'inversion' | 'scramble'
        """
        # Hyperparameters
        self.population_size   = population_size
        self.elite_size        = elite_size
        self.mutation_rate     = mutation_rate
        self.tournament_size   = tournament_size
        self.crossover_method  = crossover_method
        self.mutation_method   = mutation_method

        # Internal state
        self.map               = None
        self.num_nodes         = None
        self.population        = None

        # Results
        self.best_series       = []   # best score per generation
        self.mean_series       = []   # mean  score per generation
        self.best              = None
        self.best_path         = None
        self.fit_time          = None
        self.fitted            = False
        self.stopped_early     = False

    # ------------------------------------------------------------------ #
    #  Dunder                                                              #
    # ------------------------------------------------------------------ #
    def __str__(self):
        s  = "Genetic Algorithm Optimizer (TSP)"
        s += "\n" + "-"*40
        s += f"\nPopulation size  : {self.population_size}"
        s += f"\nElite size       : {self.elite_size}"
        s += f"\nMutation rate    : {self.mutation_rate}"
        s += f"\nTournament size  : {self.tournament_size}"
        s += f"\nCrossover method : {self.crossover_method}"
        s += f"\nMutation method  : {self.mutation_method}"
        s += "\n" + "-"*40
        s += "\nFitted: " + ("YES" if self.fitted else "NO")
        return s

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def _initialize_population(self):
        """Create population_size random permutations of [0 … n-1]."""
        pop = [np.random.permutation(self.num_nodes).tolist()
               for _ in range(self.population_size)]
        return pop

    # ------------------------------------------------------------------ #
    #  Fitness                                                             #
    # ------------------------------------------------------------------ #
    def _route_cost(self, route):
        """Total round-trip cost of a route (list of node indices)."""
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.map[route[i], route[i + 1]]
        cost += self.map[route[-1], route[0]]   # return to start
        return cost

    def _evaluate_population(self, population):
        """Return array of costs for each individual."""
        return np.array([self._route_cost(ind) for ind in population])

    # ------------------------------------------------------------------ #
    #  Selection – Tournament                                              #
    # ------------------------------------------------------------------ #
    def _tournament_select(self, population, costs, mode):
        """Pick the best (min/max) individual from a random tournament."""
        idx = np.random.choice(len(population), self.tournament_size, replace=False)
        if mode == 'min':
            winner = idx[np.argmin(costs[idx])]
        else:
            winner = idx[np.argmax(costs[idx])]
        return population[winner][:]

    # ------------------------------------------------------------------ #
    #  Crossover                                                           #
    # ------------------------------------------------------------------ #
    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX1): preserves relative order from both parents."""
        n = len(parent1)
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b+1] = parent1[a:b+1]
        fill = [x for x in parent2 if x not in child]
        pos  = list(range(b+1, n)) + list(range(a))
        for i, p in enumerate(pos):
            child[p] = fill[i]
        return child

    def _pmx_crossover(self, parent1, parent2):
        """Partially Mapped Crossover (PMX)."""
        n = len(parent1)
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b+1] = parent1[a:b+1]
        for i in range(a, b+1):
            val = parent2[i]
            if val not in child:
                pos = i
                while a <= pos <= b:
                    pos = parent2.index(parent1[pos])
                child[pos] = val
        for i in range(n):
            if child[i] == -1:
                child[i] = parent2[i]
        return child

    def _crossover(self, p1, p2):
        if self.crossover_method == 'pmx':
            return self._pmx_crossover(p1, p2)
        return self._order_crossover(p1, p2)   # default: OX

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #
    def _mutate(self, route):
        """Apply mutation with probability self.mutation_rate."""
        if np.random.random() > self.mutation_rate:
            return route
        n = len(route)
        if self.mutation_method == 'inversion':
            a, b = sorted(np.random.choice(n, 2, replace=False))
            route[a:b+1] = route[a:b+1][::-1]
        elif self.mutation_method == 'scramble':
            a, b = sorted(np.random.choice(n, 2, replace=False))
            seg = route[a:b+1]
            np.random.shuffle(seg)
            route[a:b+1] = seg
        else:   # swap (default)
            i, j = np.random.choice(n, 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    # ------------------------------------------------------------------ #
    #  Next generation                                                     #
    # ------------------------------------------------------------------ #
    def _next_generation(self, population, costs, mode):
        """Build the next generation: elitism + crossover + mutation."""
        # Sort by cost
        order = np.argsort(costs) if mode == 'min' else np.argsort(costs)[::-1]
        sorted_pop = [population[i] for i in order]

        next_gen = sorted_pop[:self.elite_size]   # elitism

        while len(next_gen) < self.population_size:
            p1 = self._tournament_select(population, costs, mode)
            p2 = self._tournament_select(population, costs, mode)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            next_gen.append(child)

        return next_gen

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(self, map_matrix, iterations=500, mode='min',
            early_stopping_count=50, verbose=True):
        """
        Run the genetic algorithm on map_matrix.

        :param map_matrix:           square cost / distance matrix (numpy ndarray)
        :param iterations:           maximum number of generations
        :param mode:                 'min' or 'max'
        :param early_stopping_count: stop if best score unchanged for this many generations
        :param verbose:              print progress
        :return:                     best score found
        """
        assert map_matrix.shape[0] == map_matrix.shape[1], "map_matrix must be square!"
        if verbose:
            print(f"Starting Genetic Algorithm with {iterations} generations "
                  f"and population size {self.population_size}…")

        self.map       = map_matrix
        self.num_nodes = map_matrix.shape[0]
        start          = time.time()

        # --- Initialise ---
        self.population = self._initialize_population()
        best_score_so_far = None
        num_equal = 0

        for gen in range(iterations):
            t0    = time.time()
            costs = self._evaluate_population(self.population)

            # Best of this generation
            gen_best_idx   = np.argmin(costs) if mode == 'min' else np.argmax(costs)
            gen_best_score = costs[gen_best_idx]
            gen_mean_score = np.mean(costs)

            self.best_series.append(gen_best_score)
            self.mean_series.append(gen_mean_score)

            # Update global best
            is_better = (
                (mode == 'min' and (best_score_so_far is None or gen_best_score < best_score_so_far)) or
                (mode == 'max' and (best_score_so_far is None or gen_best_score > best_score_so_far))
            )
            if is_better:
                best_score_so_far = gen_best_score
                self.best_path    = self.population[gen_best_idx][:]
                num_equal = 0
            else:
                num_equal += 1

            if verbose:
                print(f"Gen {gen:>4d} | best: {gen_best_score:>12,.2f} | "
                      f"overall: {best_score_so_far:>12,.2f} | "
                      f"mean: {gen_mean_score:>12,.2f} | "
                      f"({time.time()-t0:.1f}s)")

            if num_equal >= early_stopping_count:
                self.stopped_early = True
                print(f"Early stopping: {early_stopping_count} generations without improvement.")
                break

            self.population = self._next_generation(self.population, costs, mode)

        self.fit_time = round(time.time() - start)
        self.fitted   = True
        self.best     = best_score_so_far

        if verbose:
            print(f"\nGA fitted.  Runtime: {self.fit_time // 60}m {self.fit_time % 60}s. "
                  f"Best score: {self.best:,.2f}")
        return self.best

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #
    def plot(self):
        """Plot best and mean score per generation."""
        if not self.fitted:
            print("Optimizer not fitted yet – nothing to plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        generations = range(len(self.best_series))
        ax.plot(generations, self.best_series, label="Best score", color="crimson", linewidth=2)
        ax.plot(generations, self.mean_series,  label="Mean score", color="steelblue",
                linewidth=1.2, alpha=0.7, linestyle='--')
        ax.fill_between(generations, self.best_series, self.mean_series,
                        alpha=0.08, color="gray")

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Total cost (MXN)", fontsize=12)
        ax.set_title(f"Genetic Algorithm – TSP\nBest: ${self.best:,.2f} MXN", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        info = (f"Population : {self.population_size}\n"
                f"Elite      : {self.elite_size}\n"
                f"Mut. rate  : {self.mutation_rate}\n"
                f"Tournament : {self.tournament_size}\n"
                f"Crossover  : {self.crossover_method}\n"
                f"Mutation   : {self.mutation_method}\n"
                f"Fit time   : {self.fit_time // 60}m{self.fit_time % 60}s"
                + ("\n[Stopped early]" if self.stopped_early else ""))
        ax.text(0.78, 0.97, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=6))

        plt.tight_layout()
        plt.show()
