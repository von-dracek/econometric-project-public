from typing import NamedTuple

import numpy as np
import pandas as pd
from geopy import distance


class GPSPoint(NamedTuple):
    lat: float
    long: float

    def to_tuple(self):
        return (self.lat, self.long)


def _calc_distance_in_km(point1: GPSPoint, point2: GPSPoint):
    if not isinstance(point1, GPSPoint) or not isinstance(point2, GPSPoint):
        return np.nan
    return distance.distance(point1.to_tuple(), point2.to_tuple()).km


def _row_to_points(row: pd.Series):
    point1 = GPSPoint(row["latitude_x"], row["longitude_x"])
    point2 = GPSPoint(row["latitude_y"], row["longitude_y"])
    return point1, point2


def calc_distance_from_gps_coords_in_km(row: pd.Series):
    points = _row_to_points(row)
    distance = _calc_distance_in_km(*points)
    return distance
