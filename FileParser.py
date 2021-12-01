# This script is responsible for generating the ranking files given a list of species,
# eps and ratio lists

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os

from sklearn.cluster import DBSCAN

import EBirdUtil

@dataclass
class Entry:
    time: datetime
    order: float
    count: int
    latitude: float
    longitude: float
    distance: float


def get_birds(filename):
    data = []
    with open("birds/" + filename) as file:
        for line in file:
            line_arr = line.split("\t")
            if line_arr[28] == "":
                time = datetime.strptime(line_arr[27], "%Y-%m-%d")
            else:
                time = datetime.strptime(line_arr[27] + " " + line_arr[28], '%Y-%m-%d %H:%M:%S')

            order = float(line_arr[2])
            count = 1 if line_arr[8] == "X" else line_arr[8]
            latitude = line_arr[25]
            longitude = line_arr[26]
            distance = 0 if line_arr[35] == "" else float(line_arr[35])
            # print("order: ", order, "count: ", count, "lat: ", latitude, "long: ", longitude, "distance: ", distance, "time: ", time)

            data.append(Entry(time, order, int(count), float(latitude), float(longitude), distance))

    return data


SECONDS_IN_YR = 31536000
SECONDS_IN_DAY = 86400
MILES_IN_DEGREE = 68.7


def get_unclustered_ranks(species):
    count_dict = {}
    ranking = []
    for bird in species:
        bird_count = 0
        data = get_birds(bird + ".txt")
        for entry in data:
            bird_count += 1
        count_dict[bird] = bird_count
    count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
    for bird in count_dict.keys():
        ranking.append(bird)
    return ranking


def generate_dbscann_ranked_lists(species, eps_list, ratio_list):
    for filename in os.listdir("ranking_test/results/"):
        if filename.endswith("txt"):
            os.remove("ranking_test/results/" + filename)

    for eps in eps_list:
        for ratio in ratio_list:
            with open("ranking_test/results/" + str(round(eps, 1)) + "_" + str(round(ratio, 1)) + ".txt",
                      "w+") as out_file:
                count_dict = {}
                for bird in species:
                    data = get_birds(bird + ".txt")

                    points = []
                    for entry in data:
                        # Days since 1970
                        time_val = entry.time.timestamp() / SECONDS_IN_DAY
                        # Days/mile
                        days_per_miles = ratio / eps
                        # Time Val measured in miles
                        time_val = time_val / days_per_miles

                        # Get distance of lat, long point in miles from 0,0 (for more accurate distance clustering)
                        lat_dist = EBirdUtil.geo_distance(entry.latitude, entry.longitude, 0, entry.latitude)
                        lon_dist = EBirdUtil.geo_distance(entry.latitude, entry.longitude, entry.latitude, 0)

                        points.append([lat_dist, lon_dist, time_val])
                    if len(points) == 0:
                        continue

                    db = DBSCAN(eps=eps, min_samples=1).fit(points)
                    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                    core_samples_mask[db.core_sample_indices_] = True
                    labels = db.labels_

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                    count_dict[bird] = n_clusters

                count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
                out_file.write("eps=" + str(eps) + " days/eps=" + str(ratio) + "\n\n")

                for i in range(len(count_dict)):
                    item = list(count_dict.items())[i]
                    out_file.write(str(i+1) + ". " + item[0] + " " + str(item[1]) + "\n")


def get_ranked_lists_from_file():
    rank_dict = {}
    file_count = 0
    for filename in os.listdir("ranking_test/results/"):
        if not filename.endswith("txt"):
            continue

        file_count += 1
        with open("ranking_test/results/" + filename) as in_file:
            name_arr = filename[:len(filename) - 4].split('_')
            eps = float(name_arr[0])
            ratio = float(name_arr[1])

            line_num = 0
            ranking = []

            for line in in_file:
                species_name = ""
                if line_num <= 1:
                    line_num += 1
                    continue

                line_arr = line.split(' ')
                for i in range(1, len(line_arr) - 1):
                    species_name += line_arr[i]
                    if i != len(line_arr) - 2:
                        species_name += " "

                ranking.append(species_name)
        rank_dict[(eps, ratio)] = ranking
    rank_dict = dict(sorted(rank_dict.items(), key=lambda x: (x[0][0], x[0][1]), reverse=False))
    # print(rank_dict)
    # print("Files: ", file_count)
    return rank_dict

def find_most_common_ranking():
    ranking_frequency = {}
    file_count = 0
    for filename in os.listdir("ranking_test/results/"):
        if not filename.endswith("txt"):
            continue

        file_count += 1
        with open("ranking_test/results/" + filename) as in_file:
            line_num = 0
            ranking = []

            for line in in_file:
                species_name = ""
                if line_num <= 1:
                    line_num += 1
                    continue

                line_arr = line.split(' ')
                for i in range(1, len(line_arr) - 1):
                    species_name += line_arr[i]
                    if i != len(line_arr) - 2:
                        species_name += " "

                ranking.append(species_name)

            ranking = tuple(ranking)
            if ranking in ranking_frequency.keys():
                ranking_frequency[ranking] += 1
            else:
                ranking_frequency[ranking] = 1

    max = 0
    most_common = []
    for ranking, freq in ranking_frequency.items():
        # print(ranking, freq)
        if freq > max:
            max = freq
            most_common = ranking
    return list(most_common)