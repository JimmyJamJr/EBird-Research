from dataclasses import dataclass
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import numpy
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


def geo_to_graph(xyz, map: Basemap):
    print(xyz)
    coords = []
    for coord in xyz:
        x, y = map(coord[1], coord[0])
        coords.append([x, y, coord[2]])

    return numpy.array(coords)


filename = "Lesser Black-backed Gull.txt"
data = FileParser.get_birds(filename)

auto_bounds = False
# lat_bounds = [35.75, 42.25]
lat_bounds = [38.3, 40.5]
# long_bounds = [-120.25, -113.75]
long_bounds = [-120.25, -119]

SECONDS_IN_YR = 31536000
SECONDS_IN_DAY = 86400
MILES_IN_DEGREE = 68.7
eps = 6
time_to_space_ratios = 8

points = []
pltPoints = []
for entry in data:
    time_of_year = entry.time - entry.time.replace(month=1, day=1, hour=0, minute=0, microsecond=0)
    # Interpolate the time value between min and max
    time_val = (entry.time.timestamp() / SECONDS_IN_DAY) / (time_to_space_ratios * eps) / MILES_IN_DEGREE
    if entry.time.year >= 2015:
        points.append([entry.latitude, entry.longitude, time_val])
        pltPoints.append([entry.latitude, entry.longitude, mdates.date2num(entry.time)])
        # print(time_of_year)
        # print(time_val)

db = DBSCAN(eps = eps / MILES_IN_DEGREE, min_samples=1).fit(points)
pred_y = db.fit_predict(points)
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


# Visualization

points = numpy.array(pltPoints)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.zaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.zaxis.set_major_locator(mdates.MonthLocator(interval=3))

if not auto_bounds:
    ax.set_xlim3d(lat_bounds[0], lat_bounds[1])
    ax.set_ylim3d(long_bounds[0], long_bounds[1])
ax.set_zlim3d([date(2015, 1, 1), date(2022, 1, 1)])

map = Basemap(fix_aspect=False, llcrnrlon=long_bounds[0], llcrnrlat=lat_bounds[0], urcrnrlon=long_bounds[1], urcrnrlat=lat_bounds[1], resolution='i', ax=ax, projection='cyl')
ax.add_collection3d(map.drawstates(linewidth=0.5, ax=ax), zs=ax.get_zlim3d()[0])
polys = []
for polygon in map.lakepolygons:
    polys.append(polygon.get_coords())
lc = PolyCollection(polys, edgecolor='blue', facecolor='blue', closed=False)
ax.add_collection3d(lc, zs=ax.get_zlim3d()[0])
ax.add_collection3d(map.readshapefile('Nevada_Counties/NVCountyBoundaries', "counties", color='grey', linewidth=.3)[-1], zs=ax.get_zlim3d()[0])
map.drawparallels(range(35, 42, 1), linewidth=1, dashes=[2, 2], labels=[0,0,1,0], ax=ax)

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Date')
ax.view_init(azim=200)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, .05]

    class_member_mask = (labels == k)
    xyz = geo_to_graph(points[class_member_mask & core_samples_mask], map)
    # print(xyz)
    if len(xyz) > 0:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    xyz = points[class_member_mask & ~core_samples_mask]
    if len(xyz) > 0:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=0)

plt.show()
plt.ion()



