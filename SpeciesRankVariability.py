from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import FileParser
import statistics
from scipy import stats

eps_list = np.linspace(.25, 15, 60)
ratio_list = np.linspace(.25, 15, 60)
species = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

if input("Regnerate Ranks? ") == "y":
    print("Regenerating rank files...")
    FileParser.generate_dbscann_ranked_lists(species, eps_list, ratio_list)

ranking_dict = FileParser.get_ranked_lists_from_file()
matrix = []
eps_checked = []
for coords, ranks in ranking_dict.items():
    ratio_checked = []
    if not coords[0] in eps_checked:
        eps_checked.append(coords[0])
        matrix.append([])
    if not coords[1] in ratio_checked:
        ratio_checked.append(coords[1])
        matrix[eps_checked.index(coords[0])].append(ranks)

matrix = np.array(matrix)

species_ranks = {}
for bird in species:
    species_ranks[bird] = []

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        for k in range(len(matrix[i][j])):
            species_ranks[matrix[i][j][k]].append(k + 1)

med_dict = {}
std_dev_dict = {}
mad_dict = {}

for bird, ranks in species_ranks.items():
    med_dict[bird] = statistics.median(ranks)
    std_dev_dict[bird] = statistics.stdev(ranks)
    mad_dict[bird] = stats.median_abs_deviation(ranks)

# print(med_dict)
# print(std_dev_dict)
# print(mad_dict)

for bird in species:
    print("{}: Med = {}, Std Dev = {:.2f}, MAD = {}".format(bird, med_dict[bird], std_dev_dict[bird], mad_dict[bird]))
