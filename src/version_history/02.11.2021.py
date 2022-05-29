import tensorflow as tf
import numpy as np
import sklearn as skl
from sklearn import linear_model, datasets
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import csv


def main(version: bool = False):
    if version:
        print(f"TF Version: {tf.__version__}")
        print(f"NP Version: {np.__version__}")
        print(f"SKL Version: {skl.__version__}")
        print(f"PD Version: {pd.__version__}")


def plot(count):
    x, y = [], []

    with open("data_sets/COMBO17.csv", "r") as COMBO17:
        plots = csv.reader(COMBO17, delimiter =',')



        c=1
        for row in plots:
            if c==1:
                c+=1
            elif c<=count and c!=1:
                x.append(float(row[1]))
                y.append(float(row[12]))
                c+=1
            else:
                break

        plt.scatter(x, y, 1)
        plt.xlabel('red-magnitude')
        plt.ylabel('redshift')
        plt.show()


if __name__ == "__main__":
    # main(version=True)
    plot(count=9997)
