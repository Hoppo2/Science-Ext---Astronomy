# from astroquery.mast import Observations, Catalogs, Tesscut, Zcut
# from astropy.coordinates import SkyCoord

import xlsxwriter
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer


class _Data:
    """
    This class (_Data) constructs the arrays for various compilations of data
    """

    def __init__(self):
        self.data = pd.read_csv("data_sets/COMBO17.csv")
        self.Incl_Mcz = np.array(
            list(zip(self.data["Mcz"], self.data["W420FE"], self.data["W462FE"], self.data["W485FD"],
                     self.data["W518FE"], self.data["W571FS"], self.data["W604FE"], self.data["W646FD"],
                     self.data["W696FE"], self.data["W753FE"], self.data["W815FS"], self.data["W856FD"],
                     self.data["W914FD"]
                     )))
        self.Excl_Mcz = np.array(
            list(zip(self.data["W420FE"], self.data["W462FE"], self.data["W485FD"],
                     self.data["W518FE"], self.data["W571FS"], self.data["W604FE"], self.data["W646FD"],
                     self.data["W696FE"], self.data["W753FE"], self.data["W815FS"], self.data["W856FD"],
                     self.data["W914FD"]
                     )))
        self.Intensity_Mag = np.array(
            list(zip(self.data["Mcz"], self.data["mumax"]
                     )))


class Regression(_Data):
    def __init__(self):
        _Data.__init__(self)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Excl_Mcz,
            self.data["Mcz"],
            test_size=0.2,
            random_state=5
        )

    def polynomial(self, degree, include_bias=False):
        """
        This is not properly working #TODO: get coefficients of polynomial regression line
        :param degree: Degree of the polynomial regression line
        :param include_bias: #TODO: Don't know what this does
        :return: type:list -> [regression model, regression score]
        """
        reg = PolynomialFeatures(degree=degree, include_bias=include_bias)
        reg.fit_transform(self.X_train, self.y_train)

        score = None

        return [reg, score]

    def linear(self):
        """
        Runs a linear regression analysis. No significance given by score.
        :return: type:list -> [regression model, regression score]
        """
        reg = LinearRegression().fit(self.X_train, self.y_train)

        score = reg.score(self.X_test, self.y_test)

        return [reg, score]


class Cluster(_Data):
    def __init__(self):
        _Data.__init__(self)

    def cluster_Excl_Mcz(self, n):
        """
        This method creates a cluster analysis excluding the redshift variable
            7-8 is the optimum grouping
        :param n: Number of CLuster Groups
        :return: None
        """
        model = KMeans(n_clusters=n)
        model.fit(self.Excl_Mcz)
        silhouette = silhouette_score(self.Excl_Mcz, model.labels_)
        inertia = model.inertia_
        seed = model.random_state
        return [model, silhouette, inertia, seed]

    def cluster_Incl_Mcz(self, n):
        """
        This method creates a cluster analysis including the redshift variable
            3-4 is the optimum grouping
        :param n: Number of CLuster Groups
        :return: None
        """
        model = KMeans(n_clusters=n)
        model.fit(self.Incl_Mcz)
        silhouette = silhouette_score(X=self.Incl_Mcz, labels=model.labels_)
        inertia = model.inertia_
        seed = model.random_state
        return [model, silhouette, inertia, seed]

    def visualiser(self, model, Include_Mcz=True):
        """
        This method creates a visualiser plot based upon a cluster model.
        :param Include_Mcz: True: Incl_Mcz, False: Excl_Mcz
        :param model: k-means cluster model
        :return: visualiser object
        """
        if Include_Mcz:
            _data = self.Incl_Mcz
        else:
            _data = self.Excl_Mcz
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(_data)  # _data is the data array used in the creation of cluster model

        visualiser.show()

        return visualiser

    @staticmethod
    def silhouette_graph(n_iterations: int):
        """
        This graphs the silhouette score of the range of clusters between 2 and n_iterations.
        :param n_iterations: Number of iterations (k) calculated. The silhouette score of the model calculated by the
        number of clusters range (2, n_iterations) will be graphed.
        :return: plt object
        """
        inertia_score = []
        k = range(2, n_iterations)
        for n in k:
            inertia_score.append(Cluster().cluster_Incl_Mcz(n)[1])

        plt.plot(k, inertia_score)
        plt.xlabel("n clusters")
        plt.ylabel("silhouette score")
        plt.show()

        return plt

    @staticmethod
    def inertia_graph(n_iterations: int):
        """
        This graphs the inertia score of the range of clusters between 2 and n_iterations.
        :param n_iterations: Number of iterations (k) calculated. The inertia of the model calculated by the
        number of clusters range (2, n_iterations) will be graphed.
        :return: plt object
        """
        inertia_score = []
        k = range(2, n_iterations)
        for n in k:
            inertia_score.append(Cluster().cluster_Incl_Mcz(n)[2])

        plt.plot(k, inertia_score)
        plt.xlabel("n clusters")
        plt.ylabel("inertia")
        plt.show()

        return plt


class Plot:
    def __init__(self, count: int):
        self.count = count

    def rs_i(self):
        """
        To be honest, this is a pretty *badly* constructed method. Plots (two axis) luminosity in the restricted
        visible wavelengths (x axis) and redshift (y axis).
        :return: None
        """
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

            return None

    def s_i(self):
        """
        To be honest, this is also a pretty *badly* constructed method. Plots (two axis) luminosity in the complete
        visible wavelengths in the COMBO17 dataset (x axis) and redshift (y axis).
        :return: None
        """
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

            return None


class Graphical(_Data):
    def __init__(self):
        _Data.__init__(self)

    def three_band(self):
        """
        Three-dimensional graph of the relationships between wavelengths 462 (x axis), 646 (y axis), 914 (z axis) and
        redshift (colour map)
        :return: plt object
        """
        xdata = self.data["W462FE"]
        ydata = self.data["W646FD"]
        zdata = self.data["W914FD"]
        cdata = self.data["Mcz"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(x=xdata, y=ydata, z=zdata, c=cdata, cmap=plt.hot())

        fig.colorbar(img)
        ax.set_xlabel('Wavelength = 462nm')
        ax.set_ylabel('Wavelength = 646nm')
        ax.set_zlabel('Wavelength = 914nm')

        plt.show()

        return plt

    def two_axis(self):
        """
        Two-dimensional graph of the relationship between galaxy luminosity (y axis) and redshift (x axis)
        :return: plt object
        """
        xdata = self.data["Mcz"]
        ydata = self.data["mumax"]

        plt.scatter(xdata, ydata)

        plt.xlabel("Redshift (Mcz)")
        plt.ylabel("Luminosity (μmax)")

        plt.show()

        return plt


class ManualLabour:
    @staticmethod
    def best_model(n_iterations, n_clusters):  # TODO: Rename Score
        """
        Function to optimise the random_state for the cluster
        :param n_iterations: Number of initialisations of the cluster model
        :param n_clusters: Number of cluster groups in each iteration
        :return: optimum cluster model
        """
        best_model = [None, 0, 0]

        for x in range(1, n_iterations):
            if Cluster().cluster_Incl_Mcz(n_clusters)[1] > best_model[1]:
                best_model = Cluster().cluster_Incl_Mcz(n_clusters)
                print(x, "best")
            else:
                print(x)

        return best_model

    @staticmethod
    def Excel_Export(array, Workbook_Name):
        """
        Exports an array to a excel workbook named Workbook_Name
        :param array: Numpy Array
        :param Workbook_Name: Name given to the excel workbook
        :return: None
        """
        workbook = xlsxwriter.Workbook(f'{Workbook_Name}.xlsx')
        worksheet = workbook.add_worksheet()

        row = 0

        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)

        workbook.close()

        return None


if __name__ == "__main__":
    # Use this to run functions
