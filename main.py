# import numpy as np
# from SimpleBinaryGeneticAlgorithm import SimpleBinaryGenetic
#
# # The function that we want to find it's maximum
# def trigonometric_exponential_function(x: []):
#     value = (1 + np.cos(2 * np.pi * x[0] * x[1])) * np.exp(-(np.abs(x[0])+np.abs(x[1]))/2)
#     return value
#
#
# if __name__ == '__main__':
#     # Instantiating simple binary genetic algorithm object
#     sbg = SimpleBinaryGenetic(
#         n_chromosome=128,
#         m_gene=2,
#         genes_lengths=np.array([8, 8]),
#         genes_intervals=np.array([[-4, 2], [-1.5, 1]]),
#         target_function=trigonometric_exponential_function)
#
#     # Beginning execution of the algorithm
#     sbg.begin_learning(1000)
