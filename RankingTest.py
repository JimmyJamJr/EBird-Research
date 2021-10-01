from dataclasses import dataclass
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy
import numpy as np

from sklearn.cluster import DBSCAN

import FileParser


@dataclass
class Entry:
    time: datetime
    order: float
    count: int
    latitude: float
    longitude: float
    distance: float

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

line_chart = False
grid = True
distances = True


SECONDS_IN_YR = 31536000
SECONDS_IN_DAY = 86400
MILES_IN_DEGREE = 68.7
eps_list = numpy.array([1., 2., 4., 8., 16., 32.])
time_to_space_ratios = numpy.array([.25, .5, 1., 2., 4., 8., 16.])

fig = plt.figure()

ranking_dict = {}
ranking_matrix = [[0 for x in range(len(time_to_space_ratios))] for y in range(len(eps_list))]
for eps in eps_list:
    ranking_dict[eps] = {}
    for ratio in time_to_space_ratios:
        with open("ranking_test/results/" + str(round(eps, 1)) + "_" + str(round(ratio, 1)) + ".txt", "w+") as out_file:
            count_dict = {}
            for filename in os.listdir("ranking_test/birds/"):
                if not filename.endswith("txt"):
                    continue

                data = FileParser.get_birds("ranking_test/birds/" + filename)

                points = []
                for entry in data:
                    time_val = (entry.time.timestamp() / SECONDS_IN_DAY) / (ratio * eps) / MILES_IN_DEGREE
                    points.append([entry.latitude, entry.longitude, time_val])

                if len(points) == 0:
                    continue

                db = DBSCAN(eps = eps / MILES_IN_DEGREE, min_samples=1).fit(points)
                pred_y = db.fit_predict(points)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                count_dict[filename[:len(filename) - 4]] = n_clusters

            count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
            # out_file.write("eps=" + str(eps) + " days/eps=" + str(ratio) + "\n\n")
            ranks = ranking_matrix[np.where(eps_list == eps)[0][0]][np.where(time_to_space_ratios == ratio)[0][0]] = []
            for i in range(len(count_dict)):
                item = list(count_dict.items())[i]
                ranks.append(item[0])
            # out_file.write(str(i+1) + ". " + item[0] + " " + str(item[1]) + "\n")

                if item[0] in ranking_dict[eps]:
                    ranking_dict[eps][item[0]].append(i+1)
                else:
                    ranking_dict[eps][item[0]] = [i+1]

print(ranking_dict)

if line_chart:
    i = 1
    for eps, dict in ranking_dict.items():
        if eps != 4.0:
            continue

        ax = fig.add_subplot(1, 1, i)
        i += 1
        for filename, rankings in dict.items():
            ax.plot(time_to_space_ratios, rankings, 'x-', label=filename)

        ax.set_xlabel('Days/EPS')
        ax.set_ylabel('Rank')
        ax.set_title("EPS = " + str(eps) + " Miles")
        ax.legend(loc=2)

test_epses = [1.0, 2.0, 4.0, 8.0]
if grid:
    for graph_index in range(4):
        birds = []
        test_eps = test_epses[graph_index]
        for name, ranks in ranking_dict[test_eps].items():
            birds.append(name)
        matrix = np.zeros((len(birds), len(birds)))

        working_dict = ranking_dict[test_eps]
        print(working_dict)
        for i in range(len(birds)):
            species_ranks = list(working_dict.values())[i]
            for j in range(len(birds)):
                comparison_ranks = list(working_dict.values())[j]
                for k in range(len(comparison_ranks)):
                    print(birds[i], ": ", species_ranks[k], " vs ", birds[j], ": ", comparison_ranks[k])
                    if species_ranks[k] > comparison_ranks[k]:
                        matrix[i][j] += 1
                    elif species_ranks[k] < comparison_ranks[k]:
                        matrix[i][j] -= 1

        print(matrix)

        ax = fig.add_subplot(2, 2, graph_index + 1)
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

        ax.set_xticks(np.arange(len(birds)))
        ax.set_yticks(np.arange(len(birds)))
        ax.set_xticklabels(birds)
        ax.set_yticklabels(birds)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(birds)):
            for j in range(len(birds)):
                text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

        ax.set_title("Species Ranking vs Ranking (EPS = " + str(test_eps) + ", higher number = more rare)")

        # fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

if distances:
    ratio_matrix = np.zeros((len(ranking_matrix), len(ranking_matrix[0])))
    eps_matrix = np.zeros((len(ranking_matrix), len(ranking_matrix[0])))

    print("Ratio Differences Matrix:")
    for r in range(len(ranking_matrix)):
        for c in range(len(ranking_matrix[r])):
            ratio_matrix[r][c] = 0 if c == 0 else normalised_kendall_tau_distance(ranking_matrix[r][c], ranking_matrix[r][c-1])
    print(ratio_matrix)

    print("\n\nEPS Differences Matrix:")
    for r in range(len(ranking_matrix)):
        for c in range(len(ranking_matrix[r])):
            eps_matrix[r][c] = 0 if r == 0 else normalised_kendall_tau_distance(ranking_matrix[r][c], ranking_matrix[r-1][c])
    print(eps_matrix)





# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

