from astroquery.mast import Observations, Catalogs, Tesscut, Zcut
from astropy.coordinates import SkyCoord

import tensorflow as tf
import sklearn as skl
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from yellowbrick.cluster import SilhouetteVisualizer

import csv
import os


class _Data:
    def __init__(self):
        self.data = pd.read_csv("data_sets/COMBO17.csv")
        self.points = np.array(
            list(zip(self.data["Mcz"], self.data["W420FE"], self.data["W462FE"], self.data["W485FD"],
                     self.data["W518FE"],
                     self.data["W571FS"], self.data["W604FE"],
                     self.data["W646FD"], self.data["W696FE"], self.data["W753FE"], self.data["W815FS"],
                     self.data["W856FD"], self.data["W914FD"],
                     self.data["W914FE"])))

        self.spoints = np.array(
            list(zip(self.data["Mcz"], self.data["UFS"], self.data["BFS"], self.data["VFD"], self.data["RFS"],
                     self.data["IFD"])))


class Regression(_Data):
    def __init__(self):
        _Data.__init__(self)


class Cluster(_Data):
    def __init__(self):
        _Data.__init__(self)

    def model(self, cluster):
        model = KMeans(n_clusters=cluster)
        model.fit(self.spoints)

        return model

    def visualiser(self, model):
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(self.spoints)

        visualiser.show()

    def three_band(self):
        xdata = self.data["W462FE"]
        ydata = self.data["W646FD"]
        zdata = self.data["W914FD"]
        cdata = self.data["Mcz"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(xdata, ydata, zdata, c=cdata, cmap=plt.hot())
        fig.colorbar(img)
        ax.set_xlabel('W462FE')
        ax.set_ylabel('W646FD')
        ax.set_zlabel('W914FD')

        plt.show()

    def two_band_rs(self):
        xdata = self.data["W462FE"]
        ydata = self.data["Mcz"]
        zdata = self.data["W914FE"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter([xdata], [ydata], [zdata])
        ax.set_xlabel('W462FE')
        ax.set_ylabel('Mcz')
        ax.set_zlabel('W914FE')

        plt.show()


if __name__ == "__main__":
    x = Cluster()
    x.three_band()
    print(x.model(cluster=2).cluster_centers_)
