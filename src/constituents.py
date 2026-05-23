"""Tidal constituents (angular speeds in degrees per hour).

Values follow the Cartwright & Edden (1973) convention used in the paper.
We include the 11 constituents from the paper, plus M4 and MS4 for shallow
water (relevant for estuarine sites like Haldia and Diamond Harbour on the
Hooghly).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constituent:
    name: str
    period_hours: float
    speed_deg_per_hour: float
    kind: str  # "diurnal" | "semidiurnal" | "long-period" | "shallow-water"
    ce: float  # tidal generating potential coefficient (paper Table 2)


CONSTITUENTS: list[Constituent] = [
    # Semidiurnal
    Constituent("M2", 12.42060120, 28.9841042, "semidiurnal", 0.90809),
    Constituent("S2", 12.00000000, 30.0000000, "semidiurnal", 0.42248),
    Constituent("N2", 12.65834751, 28.4397295, "semidiurnal", 0.17386),
    Constituent("K2", 11.96723606, 30.0821373, "semidiurnal", 0.11498),
    # Diurnal
    Constituent("K1", 23.93447213, 15.0410686, "diurnal", 0.53011),
    Constituent("O1", 25.81933871, 13.9430356, "diurnal", 0.37694),
    Constituent("P1", 24.06588766, 14.9589314, "diurnal", 0.17543),
    Constituent("Q1", 26.86835000, 13.3986609, "diurnal", 0.07217),
    # Long-period
    Constituent("Mf",  327.8599387, 1.0980331, "long-period", 0.15647),
    Constituent("Msf", 354.3670666, 1.0158958, "long-period", 0.01369),
    Constituent("Mm",  661.3111655, 0.5443747, "long-period", 0.08254),
    # Shallow-water (compound) -- important in estuaries
    Constituent("M4",  6.21030060, 57.9682084, "shallow-water", 0.0),
    Constituent("MS4", 6.10333927, 58.9841042, "shallow-water", 0.0),
    Constituent("MN4", 6.26917320, 57.4238337, "shallow-water", 0.0),
    Constituent("M6",  4.14020040, 86.9523127, "shallow-water", 0.0),
    Constituent("S4",  6.00000000, 60.0000000, "shallow-water", 0.0),
    Constituent("2N2", 12.90537297, 27.8953548, "semidiurnal", 0.02300),
    # Solar annual & semi-annual (slow seasonal forcing)
    Constituent("Sa",  8766.15265,  0.0410686, "long-period", 0.01160),
    Constituent("Ssa", 4383.07632,  0.0821373, "long-period", 0.07287),
]


def by_name(name: str) -> Constituent:
    for c in CONSTITUENTS:
        if c.name == name:
            return c
    raise KeyError(name)
