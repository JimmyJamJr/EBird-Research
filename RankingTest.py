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


SECONDS_IN_YR = 31536000
SECONDS_IN_DAY = 86400
MILES_IN_DEGREE = 68.7
eps_list = numpy.array([1., 2., 4., 8., 16., 32.])
time_to_space_ratios = numpy.array([.25, .5, 1., 2., 4., 8., 16.])

fig = plt.figure()

ranking_dict = {}
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

                count_dict[filename] = n_clusters

            count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
            out_file.write("eps=" + str(eps) + " days/eps=" + str(ratio) + "\n\n")
            for i in range(len(count_dict)):
                item = list(count_dict.items())[i]
                out_file.write(str(i+1) + ". " + item[0] + " " + str(item[1]) + "\n")

                if item[0] in ranking_dict[eps]:
                    ranking_dict[eps][item[0]].append(i+1)
                else:
                    ranking_dict[eps][item[0]] = [i+1]

print(ranking_dict)
i = 1
for eps, dict in ranking_dict.items():
    if eps != 1.0:
        continue

    ax = fig.add_subplot(1, 1, i)
    i += 1
    for filename, rankings in dict.items():
        ax.plot(time_to_space_ratios, rankings, 'x-', label=filename)

    ax.set_xlabel('Days/EPS')
    ax.set_ylabel('Rank')
    ax.set_title("EPS = " + str(eps) + " Miles")
    ax.legend(loc=2)

# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

