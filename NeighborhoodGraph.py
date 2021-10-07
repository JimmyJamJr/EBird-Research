from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import FileParser

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


def kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered

graph_option = int(input("What kind of graph? "))

eps_list = np.linspace(.25, 15, 60)
ratio_list = np.linspace(.25, 15, 60)
neighborhood_size = 5
species = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

if input("Regnerate Ranks? ") == "y":
    print("Regenerating rank files...")
    FileParser.generate_dbscann_ranked_lists(species, eps_list, ratio_list)

unclustered_ranks = []
if graph_option == 2:
    print("Unclustered: ", unclustered_ranks)
unclustered_ranks = FileParser.get_unclustered_ranks(species)
most_common_ranks = []
if graph_option == 3:
    most_common_ranks = FileParser.find_most_common_ranking()
    print("Most Common: ", most_common_ranks)

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
print(matrix)

in_range_coords = []
diff_matrix = np.zeros((len(matrix), len(matrix[0])))
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if (graph_option == 1):
            max_diff = 0
            for i1 in range(i-2, i+3, 1):
                if i1 >= 0 and i1 < len(matrix):
                    for j1 in range(j-2, j+3, 1):
                        if j1 >= 0 and j1 < len(matrix[i]):
                            if math.dist([eps_list[i], ratio_list[j]], [eps_list[i1], ratio_list[j1]]) < neighborhood_size:
                                diff = kendall_tau_distance(matrix[i][j], matrix[i1][j1])
                                max_diff = diff if diff > max_diff else max_diff
            diff_matrix[i][j] = max_diff
        elif (graph_option == 2):
            diff_matrix[i][j] = kendall_tau_distance(matrix[i][j], unclustered_ranks)
        elif (graph_option == 3):
            diff_matrix[i][j] = kendall_tau_distance(matrix[i][j], most_common_ranks)

print(diff_matrix)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(diff_matrix.T, cmap="Blues", interpolation="none", origin='lower')

for i in range(len(diff_matrix.T)):
    for j in range(len(diff_matrix.T[0])):
        text = ax.text(j, i, int((diff_matrix.T)[i, j]), ha="center", va="center", color="black")

ax.set_xticks(np.arange(len(eps_list)))
ax.set_yticks(np.arange(len(ratio_list)))
print(eps_list)
print(ratio_list)
ax.set_xticklabels(eps_list)
ax.set_yticklabels(ratio_list)
ax.set_xlabel('EPS')
ax.set_ylabel('Days/EPS')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
if graph_option == 1:
    ax.set_title("Max Neighborhood Diff (Neighborhood Radius: " + str(neighborhood_size) + " )")
elif graph_option == 2:
    ax.set_title("VS Unclustered Ranking Diff")
elif graph_option == 3:
    ax.set_title("VS Most Common Ranking Diff\n" + str(most_common_ranks))

plt.show()
