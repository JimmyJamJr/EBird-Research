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

test_eps = 4.0
species = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

ranking_list = []
ratio_list = []
label_list = []
diff_list = []
ranking_dict = FileParser.get_ranked_lists_from_file()
for coords, ranks in ranking_dict.items():
    if coords[0] == test_eps:
        ranking_list.append(ranks)
        ratio_list.append(coords[1])

for i in range(1, len(ranking_list)):
    diff = kendall_tau_distance(ranking_list[i], ranking_list[i-1])
    label_list.append(str(ratio_list[i-1]) + "-" + str(ratio_list[i]))
    diff_list.append(diff)

print(diff_list)
print(label_list)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(label_list, diff_list, 'xb')
ax.set_xlabel('Change in Days/EPS')
ax.set_ylabel('Diff')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("EPS = " + str(test_eps) + " Difference in Ranks for Time Ratio Change")
plt.show()
