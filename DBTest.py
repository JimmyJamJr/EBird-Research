from dataclasses import dataclass
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
from datetime import timezone

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


def lerp(min, max, val):
    return min + val * (max - min)


filename = "Sabines Gull.txt"
data = FileParser.get_birds(filename)


SECONDS_IN_YR = 31536000
SECONDS_IN_DAY = 86400
MILES_IN_DEGREE = 68.7
eps = [4., 6., 12.]

fig = plt.figure()

for i in range(3):
    time_to_space_ratios = [.25, .5, 1, 2, 4, 8, 12, 24]
    num_clusters = []

    for ratio in time_to_space_ratios:
        points = []
        for entry in data:
            # Interpolate the time value between min and max
            time_val = (entry.time.timestamp() / SECONDS_IN_DAY) / (ratio * eps[i]) / MILES_IN_DEGREE
            if entry.time.year >= 2015:
                points.append([entry.latitude, entry.longitude, time_val])

        db = DBSCAN(eps = eps[i] / MILES_IN_DEGREE, min_samples=1).fit(points)
        pred_y = db.fit_predict(points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        num_clusters.append(n_clusters)
        # print("Estimated # of Clusters: ", n_clusters)
        # print("Estimated # of Singletons: ", n_noise)

    ax = fig.add_subplot(3, 1, i + 1)
    ax.plot(time_to_space_ratios, num_clusters, 'xb-')
    ax.set_xlabel('Days/EPS')
    ax.set_ylabel('Cluster Count')
    ax.set_title("EPS = " + str(eps[i]) + " Miles")

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()