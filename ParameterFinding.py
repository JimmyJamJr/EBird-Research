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


# target_ranking = [
#     "Gray Catbird", "Sabines Gull", "Mew Gull", "Lesser Black-backed Gull", "Brown Thrasher", "Lark Bunting", "Long-tailed Duck", "Acorn Woodpecker", "Eastern Phoebe", "Parasitic Jaeger", "Red Phalarope", "Long-tailed Jaeger", "Ruff", "Groove-billed Ani", "Red-faced Warbler", "Huttons Vireo", "Pomarine Jaeger"
# ]

target_ranking = [
    "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
]

ranking_dict = FileParser.get_ranked_lists_from_file()

resulting_coords = list(ranking_dict.keys())[0]
resulting_ranks = list(ranking_dict.values())[0]
min_dist = 999999
for coords, ranks in ranking_dict.items():
    dist = normalised_kendall_tau_distance(ranks, target_ranking)
    if dist < min_dist:
        resulting_coords = coords
        resulting_ranks = ranks
        min_dist = dist

print(resulting_coords)
print(resulting_ranks)
print("KT distance: ", min_dist)
