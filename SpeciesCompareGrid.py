from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


eps_list = np.linspace(.25, 30, 120)
ratio_list = np.linspace(.25, 30, 120)
species = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

if input("Regnerate Ranks? ") == "y":
    print("Regenerating rank files...")
    FileParser.generate_dbscann_ranked_lists(species, eps_list, ratio_list)

ranking_dict = FileParser.get_ranked_lists_from_file()
unclustered = list(FileParser.get_unclustered_ranks(species))
print(unclustered)
rarer_unclustered = ""

for speciesA in species:
    for speciesB in species:
        if speciesA == speciesB: continue

        if unclustered.index(speciesA) > unclustered.index(speciesB):
            rarer_unclustered = speciesA
        else:
            rarer_unclustered = speciesB

        matrix = []
        eps_checked = []
        for coords, ranks in ranking_dict.items():
            if not coords[0] in eps_checked:
                eps_checked.append(coords[0])
                matrix.append([])
            #     ratio_checked = []
            # if not coords[1] in ratio_checked:
            #     ratio_checked.append(coords[1])
            matrix[eps_checked.index(coords[0])].append(ranks)

        matrix = np.array(matrix)

        diff_matrix = np.zeros((len(matrix), len(matrix[0])))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if np.where(matrix[i][j] == speciesA)[0][0] > np.where(matrix[i][j] == speciesB)[0][0]:
                    diff_matrix[i][j] = 1
                else:
                    diff_matrix[i][j] = 0

        fig = plt.figure()
        fig.set_size_inches(12, 12)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(diff_matrix.T, cmap="cividis", interpolation="none", origin='lower', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(eps_list)))
        ax.set_yticks(np.arange(len(ratio_list)))
        ax.set_xticklabels(eps_list, fontsize=8, rotation=45)
        ax.set_yticklabels(ratio_list, fontsize=8)
        ax.set_xlabel('eps (miles)', fontsize=16)
        ax.set_ylabel('teps (days)', fontsize=16)

        plt.suptitle("{} vs {} Rarity".format(speciesA, speciesB), y=0.99, fontsize=24)
        plt.title("{} is more rare when unclustered".format(rarer_unclustered), fontsize=16)


        yellow_patch = mpatches.Patch(color='yellow', label='{} is more rare'.format(speciesA))
        blue_patch = mpatches.Patch(color='blue', label='{} is more rare'.format(speciesB))
        plt.legend(handles=[yellow_patch, blue_patch], bbox_to_anchor=(0, 1), loc='upper left', ncol=1, fontsize=16)
        plt.setp(ax.get_xticklabels()[1::2], visible=False)
        plt.setp(ax.get_yticklabels()[1::2], visible=False)
        fig.tight_layout()

        if not os.path.isdir("species_compare_graphs/{}".format(speciesA)):
            os.mkdir("species_compare_graphs/{}".format(speciesA))

        plt.savefig("species_compare_graphs/{}/{} vs {}.png".format(speciesA, speciesA, speciesB), dpi=100)
        plt.close()
