#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:47:12 2018

@author: josephpalombo
"""

import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def old_way(box_dict: np.ndarray) -> np.ndarray:
    with open("data/nfl_scores.csv", "r") as score_file:
        score_read = csv.reader(score_file, delimiter=",")
        for row in score_read:
            box_num1 = int(row[2]) % 10
            box_num2 = int(row[3]) % 10
            count = int(row[6])
            box_dict[box_num1][box_num2] += count

    box_dict = box_dict / np.sum(sum(box_dict)) * 100
    return box_dict


def new_way(box_dict: np.ndarray) -> np.ndarray:
    df = pd.read_csv(
        "data/nfl_scores.csv",
        header=None,
        names=[
            "Number",
            "Score",
            "Home",
            "Away",
            "Total",
            "Differential",
            "Count",
            "All Games",
            "Matchup",
        ],
    )
    df["Home_End"] = df["Home"] % 10
    df["Away_End"] = df["Away"] % 10
    by_end_df = (
        df[["Home_End", "Away_End", "Count"]].groupby(["Home_End", "Away_End"]).sum()
    )
    percent_df = by_end_df / by_end_df.sum() * 100.0  # type: ignore
    return percent_df.to_numpy().reshape(box_dict.shape)


def plot(scores: np.ndarray):
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


if __name__ == "__main__":
    box_dict = np.zeros((10, 10))
    s1 = time.time()
    b1 = old_way(box_dict)
    d1 = time.time() - s1
    box_dict = np.zeros((10, 10))
    s2 = time.time()
    b2 = new_way(box_dict)
    d2 = time.time() - s2
    print(f"Max {np.argmax(b2)}")
    print(f"{b1=}\n{b2=}\n{np.all(b1 == b2)=}\ntime old: {d1}\ntime new: {d2}")
    plot(b2.T)
