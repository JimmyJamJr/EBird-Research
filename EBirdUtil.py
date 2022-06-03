import collections
from math import radians, cos, sin, asin, sqrt
import os
from dataclasses import dataclass
from datetime import datetime
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import numpy as np

# Constants
SECONDS_PER_YR = 31536000
SECONDS_PER_DAY = 86400
MILES_PER_DEGREE = 68.7


# Function used to calculate the distance in miles between two points in geo coordinates
def geo_distance(lat1, lon1, lat2, lon2):
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * asin(sqrt(a))

    # Radius of earth in miles
    r = 3956

    # calculate the result
    return(c * r)


# Remove punctuation from string
def clean_name(name):
    name = name.replace("/", "-")
    name = name.replace("\'", "")
    return name


# Separate species into individual files
def separate_species_data(filename, bird_folder, replace_existing, match_list=[]):
    print("Generating individual bird files...")
    # Delete all existing bird files if replace is on
    if replace_existing:
        for file in os.scandir(os.path.join(os.getcwd(), bird_folder)):
            os.remove(file.path)
        print("Deleted existing bird files.")

    # Dictionary to store logging info
    species_dict = collections.defaultdict(lambda: 0)

    line_count = 0
    with open(filename) as in_file:
        next(in_file)

        # iterate through file
        for line in in_file:
            # iterate through line
            line_arr = line.split("\t")

            # Check with match list
            if len(match_list) == 0 or line_arr[4] in match_list:
                # Add to species count dictionary
                species = clean_name(line_arr[4])
                species_dict[species] += 1

                # Open species file and write observation
                out_file_name = os.path.join(os.getcwd(), bird_folder, species + ".txt")
                with open(out_file_name, "a+") as out_file:
                    out_file.write(line)

    print("Split finished, results: ", species_dict)


# Get a list of species that only appear in one checklist
def get_single_obs_species(filename):
    # Dictionary to store logging info
    species_dict = collections.defaultdict(lambda: 0)

    with open(filename) as in_file:
        singles = []
        next(in_file)

        # iterate through file
        for line in in_file:
            # iterate through line
            line_arr = line.split("\t")

            # Add to species count dictionary
            species = clean_name(line_arr[4])
            species_dict[species] += 1

        for species, count in species_dict.items():
            if count == 1:
                singles.append(species)

        return singles


# Given the name of a species file, return all its observations in list form
def get_species_obs(species, birds_folder="birds/"):

    @dataclass
    class Observation:
        time: datetime
        order: float
        count: int
        latitude: float
        longitude: float
        distance: float

    observations = []
    with open(birds_folder + species + ".txt") as file:
        for line in file:
            line_arr = line.split("\t")
            if line_arr[28] == "":
                time = datetime.strptime(line_arr[27], "%Y-%m-%d")
            else:
                time = datetime.strptime(line_arr[27] + " " + line_arr[28], '%Y-%m-%d %H:%M:%S')

            order = float(line_arr[2])

            # If observation count is not given, set to 1
            count = 1 if line_arr[8] == "X" else int(line_arr[8])
            latitude = line_arr[25]
            longitude = line_arr[26]
            distance = 0 if line_arr[35] == "" else float(line_arr[35])
            # print("order: ", order, "count: ", count, "lat: ", latitude, "long: ", longitude, "distance: ", distance, "time: ", time)

            observations.append(Observation(time, order, int(count), float(latitude), float(longitude), distance))
    return observations


# Get the rarity ranking given a dictionary of species and their counts
def ranking_from_counts(count_dict : dict):
    ranking = collections.defaultdict(lambda : [])
    # Code to handle ties
    current_rank = 0
    tie_count = 1
    for i in range(len(count_dict)):
        if i > 0 and list(count_dict.values())[i] == list(count_dict.values())[i-1]:
            ranking[current_rank].append( (list(count_dict.keys())[i], list(count_dict.values())[i]) )
            tie_count += 1
        else:
            current_rank += tie_count
            tie_count = 1
            ranking[current_rank].append( (list(count_dict.keys())[i], list(count_dict.values())[i]) )

    return ranking


# Get the ranking of given list of species with no clustering (based on checklist count)
def get_unclustered_ranking(species_list):
    count_dict = {}
    for bird in species_list:
        # Number of checklists the species appear in
        checklist_count = 0
        # Number of total individual observations
        observation_count = 0

        observations = get_species_obs(bird)
        for obs in observations:
            checklist_count += 1
            observation_count += obs.count

        # count_dict[bird] = (checklist_count, observation_count) # Observation count not used for now
        count_dict[bird] = checklist_count
    count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))

    return ranking_from_counts(count_dict)


# Get a ranking of a given list of species after applying dbscan clustering with parameters
def get_dbscan_ranking(species_list, eps, teps):
    count_dict = {}
    ranking = collections.defaultdict(lambda : [])
    for bird in species_list:
        observations = get_species_obs(bird)
        # Coordinate (x, y, t) of each observation in miles
        points = []
        # Number of total individual observations
        observation_count = 0

        for obs in observations:
            # Days since 1970
            time_val = obs.time.timestamp() / SECONDS_PER_DAY
            # Convert to teps (Days/mile)
            days_per_miles = teps / eps
            # Time Val measured in miles
            time_val = time_val / days_per_miles

            # Add observation count to total count
            observation_count += obs.count

            # Get distance of lat, long point in miles from 0,0 (for more accurate distance clustering)
            lat_dist = geo_distance(obs.latitude, obs.longitude, 0, obs.longitude)
            lon_dist = geo_distance(obs.latitude, obs.longitude, obs.latitude, 0)

            points.append([lat_dist, lon_dist, time_val])

        if len(points) == 0:
            print("Error: No observations found for ", bird)
            return

        db = DBSCAN(eps=eps, min_samples=1).fit(points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Store number of clusters and total number of observations (tie breaker)
        # count_dict[bird] = (n_clusters, total_count) # Tiebreaker disabled for now
        count_dict[bird] = n_clusters
    count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))

    return ranking_from_counts(count_dict)


# Given a list of species, eps, and teps, generate ranked lists for all parameter combos and save them to file
def generate_ranking_files(species_list, eps_list, teps_list, save_dir="ranking_test/results/"):
    # Delete exising ranking files
    for f in os.listdir("ranking_test/results/"):
        if f.endswith("txt"):
            os.remove("ranking_test/results/" + f)

    # Iterate through parameter space, get the ranking at each point, then write to file
    for eps in eps_list:
        for teps in teps_list:
            ranking = get_dbscan_ranking(species_list, eps, teps)
            with open("ranking_test/results/" + str(round(eps, 1)) + "_" + str(round(teps, 1)) + ".txt",
                      "w+") as out_file:
                out_file.write("eps=" + str(eps) + " days/eps=" + str(teps) + "\n\n")
                for rank, birds in ranking.items():
                    for bird in birds:
                        out_file.write(str(rank) + ". " + bird[0] + " (" + str(bird[1]) + ")\n")


if __name__ == "__main__":
    # separate_species_data(filename="ebd_US-NV_relFeb-2021.txt", bird_folder="birds", replace_existing=True)

    species = [
        "Hudsonian Godwit", "Ruff", "Groove-billed Ani", "Acorn Woodpecker", "Brown Thrasher", "Eastern Phoebe", "Gray Catbird", "Huttons Vireo", "Lark Bunting", "Lesser Black-backed Gull", "Long-tailed Duck", "Long-tailed Jaeger", "Mew Gull", "Parasitic Jaeger", "Pomarine Jaeger", "Red Phalarope", "Red-faced Warbler", "Sabines Gull"
    ]
    print(get_unclustered_ranking(species))
    print(get_dbscan_ranking(species, 2, 7))
    eps_list = np.linspace(.5, 30, 60)
    ratio_list = np.linspace(.5, 30, 60)
    generate_ranking_files(species, eps_list, ratio_list)
