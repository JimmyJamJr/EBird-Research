from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import FileParser
import csv

species_count = 0

csv_matrix = []
species = []

with open("Hurtado--pairwise-comparison-matrix.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    row_count = 0
    for row in csv_reader:
        row_count += 1
        row_list = []

        if row_count == 1:
            species = row[1:]
        else:
            for i in row[1:]:
                row_list.append(str(i))
            csv_matrix.append(row_list)

ranking = []

for i in range(len(csv_matrix)):
    for j in range(len(csv_matrix[i])):
        if csv_matrix[i][j] == '1':
            common = species[i]
            rare = species[j]
            # print(common, " is more common than ", rare)
        elif csv_matrix[i][j] == '-1':
            common = species[j]
            rare = species[i]
            # print(common, " is more common than ", rare)
        else:
            continue

eps_list = np.linspace(.5, 15, 30)
ratio_list = np.linspace(.5, 15, 30)
species2 = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

print(species)
print(species2)

if input("Regnerate Ranks? ") == "y":
    print("Regenerating rank files...")
    FileParser.generate_dbscann_ranked_lists(species, eps_list, ratio_list)


def diff_to_csv(ranking):
    diff = 0
    for i in range(len(species)):
        for j in range(i+1, len(species)):
            if csv_matrix[i][j] == '1':
                common = species[i]
                rare = species[j]
            elif csv_matrix[i][j] == '-1':
                common = species[j]
                rare = species[i]
            else:
                continue

            if np.where(ranking == common) > np.where(ranking == rare):
                diff += 1
    return diff


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

in_range_coords = []
diff_matrix = np.zeros((len(matrix), len(matrix[0])))
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        diff_matrix[i][j] = diff_to_csv(matrix[i][j])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(diff_matrix.T, cmap="Blues", interpolation="none", origin='lower')

for i in range(len(diff_matrix.T)):
    for j in range(len(diff_matrix.T[0])):
        text = ax.text(j, i, int((diff_matrix.T)[i, j]), ha="center", va="center", color="black")

ax.set_xticks(np.arange(len(eps_list)))
ax.set_yticks(np.arange(len(ratio_list)))
# print(eps_list)
# print(ratio_list)
ax.set_xticklabels(eps_list)
ax.set_yticklabels(ratio_list)
ax.set_xlabel('EPS')
ax.set_ylabel('Days/EPS')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Difference to Hurtado Ranking")
plt.show()

unclustered_ranks = FileParser.get_unclustered_ranks(species)
print(unclustered_ranks)