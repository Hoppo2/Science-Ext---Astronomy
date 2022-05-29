from astroquery.mast import Observations, Catalogs, Tesscut, Zcut
from astropy.coordinates import SkyCoord

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
            list(zip(self.data["mumax"], self.data["Mcz"]
                     )))


class Regression(_Data):
    def __init__(self):
        _Data.__init__(self)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.Excl_Mcz, self.data["Mcz"], test_size=0.2,
                                                            random_state=5)

    def polynomial(self, degree, include_bias=False):
        reg = PolynomialFeatures(degree=degree, include_bias=include_bias)
        reg.fit_transform(self.X_train, self.y_train)

        score = None

        return [reg, score]

    def linear(self):
        """"""
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
        score = silhouette_score(self.Excl_Mcz, model.labels_)
        return [model, score]

    def cluster_Incl_Mcz(self, n):
        """
        This method creates a cluster analysis including the redshift variable
            3-4 is the optimum grouping
        :param n: Number of CLuster Groups
        :return: None
        """
        model = KMeans(n_clusters=n)
        model.fit(self.Incl_Mcz)
        score = silhouette_score(self.Incl_Mcz, model.labels_)
        return [model, score]

    def visualiser(self, model, Include_Mcz=False):
        """
        This method creates a visualiser plot based upon a cluster model.
        :param Include_Mcz: True: Incl_Mcz, False: Excl_Mcz
        :param model: k-means cluster model
        :return: None
        """
        if Include_Mcz:
            _data = self.Incl_Mcz
        else:
            _data = self.Excl_Mcz
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(_data)

        visualiser.show()


class Plot:
    def __init__(self, count: int):
        self.count = count

    def rs_i(self):
        """
        To be honest, this is a pretty shittily constructed method.
        :return:
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

    def s_i(self):
        """
        To be honest, this is also a pretty shittily constructed method.
        :return:
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


class Graphical(_Data):
    def __init__(self):
        _Data.__init__(self)

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


class DoinShitForMe:  # not needed any more
    @staticmethod
    def Excel_Export(array, Workbook_Name):
        workbook = xlsxwriter.Workbook('{}.xlsx'.format(Workbook_Name))
        worksheet = workbook.add_worksheet()

        row = 0

        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)

        workbook.close()


if __name__ == "__main__":
    model = Regression().polynomial(degree=3)
    print(model[0].get_feature_names_out())
