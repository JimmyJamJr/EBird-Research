from dataclasses import dataclass
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

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


filename = "ebd_US-NV_relFeb-2021.txt"
data = []
total_bird_count = 0

with open(filename) as file:
    next(file)
    entry_count = 0
    for line in file:
        if entry_count > 50000:
            break
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
            data.append(Entry(time, order, common_name, sci_name, int(count), float(latitude), float(longitude), distance))
            entry_count += 1
        except:
            print("Error on Entry: ", line)
            continue

# data.sort(key=(lambda e: e.count), reverse=True)
# for e in data:
#     print(e)
# print("\n\n\n")

birds = {}
for entry in data:
    if entry.common_name in birds:
        b: Bird = birds[entry.common_name]
        duplicate = False
        for obs in b.observations:
            if obs.time == entry.time:
                duplicate = True
                break
        if not duplicate:
            b.count += entry.count
            b.observations.append(entry)
            total_bird_count += entry.count
    else:
        birds[entry.common_name] = Bird([entry], entry.common_name, entry.count)

birds = sorted(birds.items(), key=(lambda b: b[1].count), reverse=True)

for k, v in birds:
    print(v.common_name, v.count)

print("Total birds: ", total_bird_count)
# avg_density = 0
# avg_geo_range = 0
# v: Bird
# for k, v in birds:
#     distance = 0
#     lat_range = [9999, 0]
#     long_range = [9999, 0]
#     for obs in v.observations:
#         distance += obs.distance
#         if obs.latitude < lat_range[0]:
#             lat_range[0] = obs.latitude
#         if obs.latitude > lat_range[1]:
#             lat_range[1] = obs.latitude
#         if obs.longitude < long_range[0]:
#             long_range[0] = obs.longitude
#         if obs.longitude > lat_range[1]:
#             long_range[1] = obs.longitude
#     v.density = 0 if distance == 0 else v.count / distance
#     avg_density += v.density
#     v.geo_range = lat_range[1] - lat_range[0]
#     avg_geo_range += v.geo_range
# avg_density /= len(birds)
# avg_geo_range /= len(birds)
#
# for k, v in birds:
#     if v.density == 0:
#         v.density = avg_density
#
# birds = sorted(birds, key=(lambda b: (b[1].density + 100 * b[1].geo_range)), reverse=True)
#
# avg_index = avg_density + 100 * avg_geo_range
# for k, v in birds:
#     print(k + ": ", v.count, " Density: ", v.density, " Latitude Range: ", v.geo_range, "Index: ", (v.density + 100 * v.geo_range) - avg_index)
#
# print("\nAverage Density: ", avg_density, " Average Latitude Range: ", avg_geo_range)
#
# counts = np.array([v.count for k, v in birds])
# mean = np.mean(counts)
# std = np.std(counts)
# print("Mean: ", mean, " Std: ", std)

# plt.hist(counts, bins=30, range=[0, 4000])
# plt.ylabel('Frequency')
# plt.xlabel('Count')
# plt.show()
