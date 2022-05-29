# Data-set library imports:
import pandas
from astroquery.mast import Observations, Catalogs, Tesscut, Zcut

from astropy.coordinates import SkyCoord

# Data analysis library imports:
import tensorflow as tf

import sklearn as skl
from sklearn import linear_model, datasets
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from yellowbrick.cluster import SilhouetteVisualizer

# Other Libraries
import csv
import os

class Cluster:
    def __init__(self, n_clusters: int = None):
        self.cluster = n_clusters
        self.data = pandas.read_csv("data_sets/COMBO17.csv")
        self.points = np.array(
            list(zip(self.data["W420FE"], self.data["W462FE"], self.data["W485FD"], self.data["W518FE"],
                     self.data["W571FS"], self.data["W604FE"],
                     self.data["W646FD"], self.data["W696FE"], self.data["W753FE"], self.data["W815FS"],
                     self.data["W856FD"], self.data["W914FD"],
                     self.data["W914FE"])))

    def model(self):
        model = KMeans(n_clusters=4)
        model.fit(self.points)

        return model

    def visualiser(self, model):
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(self.points)

        visualiser.show()

    def three_band(self):
        points = np.array(
            list(zip(self.data["W420FE"], self.data["W646FD"], self.data["W914FE"])))

        

class Plot:
    def __init__(self, count: int):
        self.count = count

    def rs_i(self):
        x, y = [], []
        x2, y2 = [], []
        x3, y3 = [], []
        x4, y4 = [], []
        x5, y5 = [], []

        with open("data_sets/COMBO17.csv", "r") as COMBO17:
            plots = csv.reader(COMBO17, delimiter=',')

            c = 1
            for row in plots:
                if c == 1:
                    c += 1
                elif c <= self.count and c != 1:
                    y.append(float(row[5]))
                    y2.append(float(row[5]))
                    y3.append(float(row[5]))
                    y4.append(float(row[5]))
                    y5.append(float(row[5]))

                    x.append(float(row[55]))
                    x2.append(float(row[57]))
                    x3.append(float(row[59]))
                    x4.append(float(row[61]))
                    x5.append(float(row[63]))

                    c += 1
                else:
                    break

            figure, axis = plt.subplots(3, 2)

            axis[0, 0].scatter(x, y, 1)
            axis[0, 0].set_title("UFS")

            axis[1, 0].scatter(x2, y2, 1)
            axis[1, 0].set_title("BFS")

            axis[2, 0].scatter(x3, y3, 1)
            axis[2, 0].set_title("VFD")

            axis[0, 1].scatter(x4, y4, 1)
            axis[0, 1].set_title("RFS")

            axis[1, 1].scatter(x5, y5, 1)
            axis[1, 1].set_title("IFD")

            plt.show()

    def s_i(self):
        x, y = [], []
        x2, y2 = [], []
        x3, y3 = [], []
        x4, y4 = [], []
        x5, y5 = [], []
        x6, y6 = [], []
        x7, y7 = [], []
        x8, y8 = [], []
        x9, y9 = [], []
        x10, y10 = [], []
        x11, y11 = [], []
        x12, y12 = [], []
        x13, y13 = [], []

        with open("data_sets/COMBO17.csv", "r") as COMBO17:
            plots = csv.reader(COMBO17, delimiter=',')

            c = 1
            for row in plots:
                if c == 1:
                    c += 1
                elif c <= self.count and c != 1:
                    y.append(float(row[5]))
                    y2.append(float(row[5]))
                    y3.append(float(row[5]))
                    y4.append(float(row[5]))
                    y5.append(float(row[5]))
                    y6.append(float(row[5]))
                    y7.append(float(row[5]))
                    y8.append(float(row[5]))
                    y9.append(float(row[5]))
                    y10.append(float(row[5]))
                    y11.append(float(row[5]))
                    y12.append(float(row[5]))
                    y13.append(float(row[5]))

                    x.append(float(row[29]))
                    x2.append(float(row[31]))
                    x3.append(float(row[33]))
                    x4.append(float(row[35]))
                    x5.append(float(row[37]))
                    x6.append(float(row[39]))
                    x7.append(float(row[41]))
                    x8.append(float(row[43]))
                    x9.append(float(row[45]))
                    x10.append(float(row[47]))
                    x11.append(float(row[49]))
                    x12.append(float(row[51]))
                    x13.append(float(row[53]))

                    c += 1
                else:
                    break

            figure, axis = plt.subplots(4, 4)

            axis[0, 0].scatter(x, y, 1, c="violet")
            axis[0, 0].set_title("420")

            axis[1, 0].scatter(x2, y2, 1)
            axis[1, 0].set_title("462")

            axis[2, 0].scatter(x3, y3, 1)
            axis[2, 0].set_title("485")

            axis[3, 0].scatter(x4, y4, 1)
            axis[3, 0].set_title("518")

            axis[0, 1].scatter(x5, y5, 1)
            axis[0, 1].set_title("571")

            axis[1, 1].scatter(x6, y6, 1)
            axis[1, 1].set_title("604")

            axis[2, 1].scatter(x7, y7, 1)
            axis[2, 1].set_title("646")

            axis[3, 1].scatter(x8, y8, 1)
            axis[3, 1].set_title("696")

            axis[0, 2].scatter(x9, y9, 1)
            axis[0, 2].set_title("753")

            axis[1, 2].scatter(x10, y10, 1)
            axis[1, 2].set_title("815")

            axis[2, 2].scatter(x11, y11, 1)
            axis[2, 2].set_title("856")

            axis[3, 2].scatter(x12, y12, 1)
            axis[3, 2].set_title("914FD")

            axis[1, 3].scatter(x13, y13, 1)
            axis[1, 3].set_title("914FE")

            plt.show()


if __name__ == "__main__":
    pass
