from dataclasses import dataclass
from datetime import datetime
import os

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
    with open(filename) as file:
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
            print("order: ", order, "count: ", count, "lat: ", latitude, "long: ", longitude, "distance: ", distance, "time: ", time)

            data.append(Entry(time, order, int(count), float(latitude), float(longitude), distance))

    return data
