#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:47:12 2018

@author: josephpalombo
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "nfl_scores.csv"
COL_NAMES = [
    "Number",
    "Score",
    "Home",
    "Away",
    "Total",
    "Differential",
    "Count",
    "All Games",
    "Matchup",
]

NUM_DIGITS = 10


def old_way() -> np.ndarray:
    """Open csv row by row, get last digit of each score in slots 2/3 of csv entry, add count of games with 2 digit score entry to 1 digit score pair.
    Finally normalize by divinding by total count of games
    e.g. Row had home 14 away 10 times 1000, then add 100 to (4,0) entry in array.

    Returns:
        np.ndarray: 10x10 array matching pair of last digits to number of games.
    """
    box_dict = np.zeros((NUM_DIGITS, NUM_DIGITS))
    with open(DATA_PATH) as score_file:
        score_read = csv.reader(score_file, delimiter=",")
        for row in score_read:
            box_num1 = int(row[2]) % 10
            box_num2 = int(row[3]) % 10
            count = int(row[6])
            box_dict[box_num1][box_num2] += count

    return box_dict / np.sum(sum(box_dict)) * 100


def new_way() -> np.ndarray:
    """Use pandas to open csv, then use pivot table functionality to sum and order same dataframe.

    Returns:
        np.ndarray: 10x10 array matching pair of last digits to number of games.
    """
    df = pd.read_csv(
        DATA_PATH,
        names=COL_NAMES,
        header=None,
    )
    df["Home_End"] = df["Home"] % 10
    df["Away_End"] = df["Away"] % 10
    by_end_df = (
        df[["Home_End", "Away_End", "Count"]].groupby(["Home_End", "Away_End"]).sum()
    )
    return (by_end_df.to_numpy() / by_end_df.to_numpy().sum() * 100.0).reshape(
        NUM_DIGITS, NUM_DIGITS
    )


def plot(scores: np.ndarray) -> None:
    """Plotting function, image heatap of scores (home/away) to frequency."""
    fig, ax = plt.subplots()
    im = plt.imshow(scores, interpolation="nearest")  # type: ignore
    ax.set_xlabel("Home")
    ax.set_ylabel("Away")
    ax.figure.colorbar(im, ax=ax)  # type: ignore
    ax.set_xticks(np.arange(len(scores)))
    ax.set_yticks(np.arange(len(scores)))
    for i, row in enumerate(scores):
        for j, entry in enumerate(row):
            ax.text(j, i, f"{entry:.2f}", ha="center", va="center", color="w")  # type: ignore
    ax.set_title("NFL box Scores Probability")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Main function call. Compare ways of processing data."""
    time_funcs = [old_way, new_way]
    box_dicts = []
    for func in time_funcs:
        start = time.time()
        box_dict = func()
        diff_time = time.time() - start
        box_dicts.append(box_dict)
        print(f"Max {np.argmax(box_dict)}")
        print(f"Time {diff_time}")

    print(f"Same result?: {np.all(box_dicts[0] == box_dicts[1])}\n")
    plot(box_dicts[0].T)


if __name__ == "__main__":
    main()
