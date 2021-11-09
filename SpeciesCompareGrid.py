from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import FileParser


def kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered


species1 = "Sabines Gull"
species2 = "Lesser Black-backed Gull"

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

diff_matrix = np.zeros((len(matrix), len(matrix[0])))
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if np.where(matrix[i][j] == species1) > np.where(matrix[i][j] == species2):
            diff_matrix[i][j] = 1
        else:
            diff_matrix[i][j] = 0

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(diff_matrix.T, cmap="cividis", interpolation="none", origin='lower')

ax.set_xticks(np.arange(len(eps_list)))
ax.set_yticks(np.arange(len(ratio_list)))
ax.set_xticklabels(eps_list, fontsize=6, rotation=45)
ax.set_yticklabels(ratio_list, fontsize=6)
ax.set_xlabel('EPS')
ax.set_ylabel('Days/EPS')
ax.set_title("{} vs {} ranking (Yellow means more rare)".format(species1, species2))

plt.show()
