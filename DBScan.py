from dataclasses import dataclass
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import DBSCAN

@dataclass
class Entry:
    time: datetime
    order: float
    count: int
    latitude: float
    longitude: float
    distance: float


@dataclass
class GeoRange:
    lat_min: float
    lat_max: float
    long_min: float
    long_max: float


filename = "Aberts Towhee.txt"
data = []

with open(filename) as file:
    for line in file:
        line_arr = line.split("\t")
        try:
            time = datetime.strptime(line_arr[27] + " " + line_arr[28], '%Y-%m-%d %H:%M:%S')
        except:
            continue
        order = float(line_arr[2])
        count = line_arr[8]
        latitude = line_arr[25]
        longitude = line_arr[26]
        distance = 0 if line_arr[35] == "" else float(line_arr[35])
        try:
            data.append(Entry(time, order, int(count), float(latitude), float(longitude), distance))
        except:
            print("Error on Entry: ", line)
            continue

latitudes = []
longitudes = []
for entry in data:
    latitudes.append(entry.latitude)
    longitudes.append(entry.longitude)

locations = np.column_stack((latitudes, longitudes))

db = DBSCAN(eps=0.03, min_samples=20).fit(locations)
pred_y = db.fit_predict(locations)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print("Estimated # of Clusters: ", n_clusters)
print("Estimated # of Singletons: ", n_noise)

clusters = [0] * n_clusters
for point in db.labels_:
    clusters[point] += 1

print(clusters)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = locations[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = locations[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()
# plt.xlim(36, 36.5)
# plt.ylim(-115.5, -115)



