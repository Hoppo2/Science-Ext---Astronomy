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
            list(zip(self.data["mumax"], self.data["Mcz"]
                     )))


class Cluster(_Data):
    def __init__(self):
        _Data.__init__(self)

    def model(self, n):
        model = KMeans(n_clusters=n)
        model.fit(self.points)

        return model

    def visualiser(self, model):
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(self.points)

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

    def two_axis(self):
        xdata = self.data["Mcz"]
        ydata = self.data["mumax"]

        plt.scatter(xdata, ydata)
        plt.show()


if __name__ == "__main__":
    # Cluster().two_axis()
    Cluster().visualiser(Cluster().model(3))
