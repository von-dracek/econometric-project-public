import re

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="myApp")


def _parse_from_to_interval(row: pd.Series):
    _from = row["from"]
    _to = row["to"]
    construction_age_band_possible_values = list(range(1800, 2050, 5))
    if _to is None and isinstance(_from, str):
        (age,) = re.findall(r"\d+", _from)
        age = int(age)
        if "before" in _from:
            row["to"] = age
            row["from"] = np.nan
            row["class"] = max(
                i for i in sorted(construction_age_band_possible_values) if i <= age
            )
        elif "onwards" in _from:
            row["from"] = min(i for i in sorted(construction_age_band_possible_values) if i >= age)
            row["to"] = np.nan
        else:
            raise NotImplementedError(_from)
    else:
        if pd.isna(_to):
            return pd.Series(dtype=object)
        else:
            row["class"] = max(
                i for i in sorted(construction_age_band_possible_values) if i <= int(_to)
            )
    return row


def _get_gps_from_address(address: str):
    print(f"Getting gps data for {address}")
    gps = geolocator.geocode(address)
    try:
        return {
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "type": gps.raw["type"],
            "class": gps.raw["class"],
        }
    except Exception:
        # if we cannot get gps data, return None
        pass


def _glazing_parse(x: str):
    if isinstance(x, float):
        return "UNKNOWN"
    x = x.lower()
    if "double" in x:
        return "DOUBLE"
    elif "single" in x:
        return "SINGLE"
    elif "secondary" in x:
        return "SECONDARY"
    elif "triple" in x:
        return "TRIPLE"
    else:
        return "UNKNOWN"


def _glazed_area_parse(x: str):
    if isinstance(x, float):
        return np.nan
    x = x.lower()
    if "normal" in x:
        return "NORMAL"
    elif "much more than" in x:
        return "MUCH_BETTER_THAN_NORMAL"
    elif "more than" in x:
        return "BETTER_THAN_NORMAL"
    elif "much less than" in x:
        return "MUCH_WORSE_THAN_NORMAL"
    elif "less than" in x:
        return "WORSE_THAN_NORMAL"
    else:
        return "UNKNOWN"


def _tenure_parse(x: str):
    if isinstance(x, float):
        return np.nan
    x = x.lower()
    if "owner" in x:
        return "OWNER_OCCUPIED"
    elif "rental" in x or "rented" in x:
        return "RENTAL"
    else:
        return "UNKNOWN"


def _energy_tariff_parse(x: str):
    if isinstance(x, float):
        return "UNKNOWN"
    x = x.lower()
    if "unknown" in x or "no data" in x or "invalid" in x:
        return "UNKNOWN"
    elif "dual" in x:
        return "DUAL"
    elif "single" in x:
        return "SINGLE"
    elif "standard" in x:
        return "STANDARD"
    elif "off-peak" in x:
        return "OFF_PEAK"
    else:
        return "UNKNOWN"


def _hotwater_parse(x: str):
    if isinstance(x, float):
        return np.nan
    x = x.lower()
    if "main system" in x:
        return "MAIN_SYSTEM"
    elif "electric" in x:
        return "ELECTRIC"
    elif "gas" in x:
        return "GAS"
    elif "community" in x:
        return "COMMUNITY_SCHEME"
    else:
        return "UNKNOWN"


def _main_fuel_parse(x: str):
    if isinstance(x, float):
        return "UNKNOWN"
    x = x.lower()
    if "this is for backwards compatibility only" in x:
        return "UNKNOWN"
    elif "mains gas" in x or "lpg" in x:
        return "GAS"
    elif "electricity" in x:
        return "ELECTRICITY"
    elif "oil" in x:
        return "OIL"
    elif "to be used only when there is no":
        return "NONE"
    elif "coal" in x or "anthracite" in x:
        return "COAL"
    elif "wood" in x:
        return "WOOD"
    else:
        return "UNKNOWN"


def _main_heat_description_parse(x: str):
    if isinstance(x, float):
        return "UNKNOWN"
    x = x.lower()
    if "electric" in x:
        return "ELECTRIC"
    elif "coal" in x or "anthracite" in x:
        return "COAL"
    elif "gas" in x or "lpg" in x:
        return "GAS"
    elif "oil" in x:
        return "OIL"
    elif "wood" in x:
        return "WOOD"
    elif "community" in x:
        return "COMMUNITY"
    else:
        return "UNKNOWN"


def _built_form_parse(x: str):
    if isinstance(x, float):
        return "UNKNOWN"
    x = x.lower()
    if "detached" in x:
        return "DETACHED"
    elif "enclosed" in x and "end" in x:
        return "ENCLOSED_END"
    elif "enclosed" in x and "mid" in x:
        return "ENCLOSED_MID"
    elif "end" in x:
        return "END"
    elif "mid" in x:
        return "MID"
    elif "semi" in x:
        return "SEMI_DETACHED"
    else:
        return "UNKNOWN"
