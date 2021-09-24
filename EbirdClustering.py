from dataclasses import dataclass
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

@dataclass
class Entry:
    time: datetime
    order: float
    common_name: str
    scientific_name: str
    count: int
    latitude: float
    longitude: float
    distance: float

@dataclass
class Bird:
    observations: List[Entry]
    common_name: str
    count: int
    density: float = 0
    geo_range: float = 0


filename = "Aberts Towhee.txt"
data = []
total_bird_count = 0

test_species = "Abert's Towhee"

with open(filename) as file:
    next(file)
    total_entries = 0
    for line in file:
        line_arr = line.split("\t")
        try:
            time = datetime.strptime(line_arr[27] + " " + line_arr[28], '%Y-%m-%d %H:%M:%S')
        except:
            continue
        order = float(line_arr[2])
        common_name = line_arr[4]
        sci_name = line_arr[5]
        count = line_arr[8]
        latitude = line_arr[25]
        longitude = line_arr[26]
        distance = 0 if line_arr[35] == "" else float(line_arr[35])
        try:
            total_entries += 1
            if common_name == test_species:
                data.append(Entry(time, order, common_name, sci_name, int(count), float(latitude), float(longitude), distance))
        except:
            print("Error on Entry: ", line)
            continue

lat_range = [9999, -9999]
long_range = [9999, -9999]
latitudes = []
longitudes = []
for entry in data:
    latitudes.append(entry.latitude)
    longitudes.append(entry.longitude)

# plt.scatter(latitudes, longitudes)
# plt.show()

X = np.column_stack((latitudes, longitudes))

# K-means clustering
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(np.column_stack((latitudes, longitudes)))
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(X)
# plt.scatter(X[:,0], X[:,1], c=pred_y)
# # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plt.show()

# DBSCAN
# nearest_neighbors = NearestNeighbors(n_neighbors=11)
# neighbors = nearest_neighbors.fit(X)
# distances, indices = neighbors.kneighbors(X)
# distances = np.sort(distances[:,10], axis=0)
# fig = plt.figure(figsize=(5, 5))
# plt.plot(distances)
# plt.xlabel("Points")
# plt.ylabel("Distance")
# # plt.ylim(-1, 0.01)
# plt.xlim(1500, 2000)
# plt.show()

print(len(data))

db = DBSCAN(eps=0.05, min_samples=2).fit(X)
pred_y = db.fit_predict(X)
print(pred_y)
labels = db.labels_
g = sns.scatterplot(X[:,0], X[:,1], hue=["cluster-{}".format(data) for data in labels], s=3, legend=False)
# g.get_legend().remove()
# plt.xlim(39.35, 39.40)
# plt.ylim(-118.9, -118.8)
plt.show()