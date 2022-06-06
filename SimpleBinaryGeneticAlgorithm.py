import numpy as np
import random
import matplotlib.pyplot as plt


# Simple binary genetic algorithm class
import self as self


class SimpleBinaryGenetic:

    # Initialize the class
    def __init__(self, n_chromosome: int, m_gene: int, genes_lengths: np.ndarray, genes_intervals: np.ndarray,
                 target_function, p_crossover: float = 0.9, p_mutation: float = 0.005,
                 is_maximization=True, use_linear_ranking=False, use_sigma_limited=False,
                 use_tournament_selection=False):

        # Initializing the main parameters
        self.n_chromosome = n_chromosome
        self.m_gene = m_gene
        self.genes_lengths = genes_lengths
        self.genes_intervals = genes_intervals
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.target_function = target_function
        self.is_maximization = is_maximization

        # Initializing other parameters
        self.__genes_lengths_cumsum = np.cumsum(genes_lengths)
        self.chromosome_length = np.sum(genes_lengths)
        self.fitness = np.ndarray((n_chromosome,))
        self._selection_probability = np.ndarray((n_chromosome,))
        self._probability_cumsum = np.ndarray((n_chromosome,))
        self.__average_fitness = 0
        self.__max_fitness = 0
        self.__total_iteration = 0
        self.__use_linear_ranking = use_linear_ranking
        self.__use_sigma_limited = use_sigma_limited
        self.__use_tournament_selection = use_tournament_selection

        # Create the population
        self.population = self.__create_population()

        # Evaluate population chromosomes
        self.evaluate_fitness()

    # Create the population
    def __create_population(self):
        population = np.random.randint(2, size=self.n_chromosome * self.chromosome_length)
        population = population.reshape(self.n_chromosome, self.chromosome_length)
        return population

    # Begin execution of the algorithm
    def begin_learning(self, iteration: int = 100):
        # print('begin learning')
        avg_fitnesses = []
        max_fitnesses = []
        self.__total_iteration = iteration
        for i in range(iteration):
            if self.__use_tournament_selection:
                mating_pool = self.binary_tournament(i)
            else:
                mating_pool = self.roulette_wheel()
            # print('mating_pool:', mating_pool)
            # print('population:', self.population)
            self.crossover(mating_pool)
            self.mutate()
            self.evaluate_fitness()
            avg_fitnesses.append(self.__average_fitness)
            max_fitnesses.append(self.__max_fitness)
            print('average_fitness ', self.__average_fitness)
            print('max_fitness ', self.__max_fitness)

        draw_begin = 0
        plt.plot(range(draw_begin, iteration), avg_fitnesses[draw_begin:], color='blue', label='average mean fitness')
        plt.plot(range(draw_begin, iteration), max_fitnesses[draw_begin:], color='red', label='best so far')
        plt.xlabel('iteration')
        plt.ylabel('fitness value')
        plt.legend()
        plt.show()

    # Evaluate population chromosomes
    def evaluate_fitness(self):
        decoded_chromosomes = []
        for i in range(self.n_chromosome):
            decoded_genes = []
            for j in range(self.m_gene):
                decoded_genes.append(self.decode_gene(i, j))
            decoded_chromosomes.append(decoded_genes)

        self.evaluate_target_function(np.array(decoded_chromosomes))


    # Decode genes from binary codes to the interval relative decimals
    def decode_gene(self, chromosome_index: int, gene_index: int):
        begin = 0 if gene_index == 0 else self.__genes_lengths_cumsum[gene_index - 1]
        gene = self.population[chromosome_index][begin:begin + self.genes_lengths[gene_index]]
        x_10 = 0
        x_normal = 0
        length = gene.shape[0]
        for i in range(length):
            x_10 += gene[i] * 2 ** (length - i - 1)
        x_normal = x_10 / (2 ** length - 1)
        x = self.genes_intervals[gene_index][0] \
            + x_normal * (self.genes_intervals[gene_index][1] - self.genes_intervals[gene_index][0])
        return x

    # Calculate fitness values of each chromosome for the provided function
    def evaluate_target_function(self, decoded_chromosomes: np.ndarray):
        function_values = []
        for i in range(decoded_chromosomes.shape[0]):
            f_value = self.target_function(decoded_chromosomes[i])
            function_values.append(f_value)

        self.fitness = np.array(function_values)
        if self.is_maximization and sum(1 for f in function_values if f < 0) > 0:
            self.fitness += min(function_values)
        elif not self.is_maximization:
            self.fitness = max(function_values) - self.fitness

        self.__average_fitness = np.mean(self.fitness)
        self.__max_fitness = np.max(self.fitness)

        if self.__use_sigma_limited:
            self.fitness = self.sigma_limited()
        if self.__use_linear_ranking:
            self._selection_probability, self.fitness, self.population = self.linear_ranking()
        else:
            self._selection_probability = self.fitness / np.sum(self.fitness)

        self._probability_cumsum = np.cumsum(self._selection_probability)
        # print(self.selection_probability)
        # print(self.probability_cumsum)

    # Selecting the fittest chromosomes for the mating pool
    def roulette_wheel(self):
        selected_parents = []
        # print(self.probability_cumsum)
        for i in range(self.n_chromosome):
            rand = random.uniform(0, self._probability_cumsum[-1])
            # print('rand: ', rand)
            selected_parent = 0
            for j in range(self.n_chromosome):
                if rand < self._probability_cumsum[j]:
                    selected_parent = j
                    break
            selected_parents.append(selected_parent)
        return selected_parents

    # Running simple crossover over mating pool's chromosomes and replacing new children with the old population
    def crossover(self, mating_pool: []):
        new_population = []
        while len(mating_pool) > 0:
            pointer1 = mating_pool.pop(random.randint(0, len(mating_pool) - 1))
            pointer2 = mating_pool.pop(random.randint(0, len(mating_pool) - 1))
            if self._selection_probability[pointer2] > self._selection_probability[pointer1]:
                c = pointer1
                pointer1 = pointer2
                pointer2 = c

            if random.random() < self.p_crossover:
                cut_point = random.randint(1, self.chromosome_length - 2)
                child1 = np.concatenate((self.population[pointer1][0:cut_point],
                                         self.population[pointer2][cut_point:self.chromosome_length]))
                child2 = np.concatenate((self.population[pointer2][0:cut_point],
                                         self.population[pointer1][cut_point:self.chromosome_length]))
                new_population.append(np.array(child1))
                new_population.append(np.array(child2))
            else:
                new_population.append(self.population[pointer1])
                new_population.append(self.population[pointer2])
        self.population = np.array(new_population)

    # Mutate new chromosomes randomly with the specified probability
    def mutate(self):
        for i in range(self.n_chromosome):
            if random.random() < self.p_mutation:
                self.population[i][random.randrange(0, self.chromosome_length)] ^= 1
        self.population

    # c should be in the interval (1,5]
    def sigma_limited(self, c: float = 2):
        fit_ave = np.sum(self.fitness) / self.n_chromosome

        sqr = []
        for fit in self.fitness:
            diff = fit - fit_ave
            sqr.append(diff ** 2)
        sigma = np.sqrt(np.sum(sqr) / self.n_chromosome)

        fit_scaled = self.fitness-(fit_ave-c*sigma)
        # for fit in self.fitness:
        #     fit_scaled.append(fit - (fit_ave - c * sigma))
        return fit_scaled

    # c is for calculating q = c/n and q is for max fitness, c should be in the interval [1, 2]
    def linear_ranking(self, c: float = 1.5):
        sorted_fitness = sorted(enumerate(self.fitness), key=lambda fitness: fitness[1])
        sorted_population = []
        for fit in sorted_fitness:
            sorted_population.append(self.population[fit[0]])

        q0 = 2 - c / self.n_chromosome
        q = c / self.n_chromosome
        select_probability = []
        for index, chromosome in enumerate(sorted_population):
            select_probability.append(q - (q - q0) * (index / (self.n_chromosome - 1)))
        return select_probability, np.array([x[1] for x in sorted_fitness]), np.array(sorted_population)

    def binary_tournament(self, iteration: int):
        mating_pool = []
        for i in range(self.n_chromosome):
            parents = random.choices(range(self.population.shape[0]), k=2)
            rand = random.random()
            p = 1 / (1 + np.exp(- (self.fitness[parents[0]] - self.fitness[parents[1]]) / self.T(iteration)))
            if rand <= p:
                mating_pool.append(parents[0])
            else:
                mating_pool.append(parents[1])

        return mating_pool

    def T(self, iteration: int = 0):
        return 1 - self.sigmoid(iteration)

    def sigmoid(self, x):
        return 1 / (1 + np.exp((-5*(x-self.__total_iteration))/self.__total_iteration))

# The function that we want to find it's maximum
def trigonometric_exponential_function(x: []):
    value = (1 + np.cos(2 * np.pi * x[0] * x[1])) * np.exp(-(np.abs(x[0]) + np.abs(x[1])) / 2)
    return value


if __name__ == '__main__':
    # Instantiating simple binary genetic algorithm object
    sbg = SimpleBinaryGenetic(
        n_chromosome=128,
        m_gene=2,
        genes_lengths=np.array([8, 8]),
        genes_intervals=np.array([[-4, 2], [-1.5, 1]]),
        target_function=trigonometric_exponential_function,
        use_linear_ranking=True,
        use_sigma_limited=True,
        use_tournament_selection=True
    )

    # Beginning execution of the algorithm
    sbg.begin_learning(1000)
