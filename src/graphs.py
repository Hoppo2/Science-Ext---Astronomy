
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import _Data


class Plot(_Data):
    def __init__(self, count: int):
        self.count = count
        _Data.__init__(self)

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

    def redshift_luminosity(self):
        x, y = self.Mcz_prop[0], self.Mcz_prop[8]

        plt.scatter(x, y)

        print(max(self.Mcz_prop[0]))

        plt.xlabel("Redshift (Mcz)")
        plt.ylabel("Luminosity (Rmag)")

        plt.show()

        return plt

    def three_band_test(self):
        """
        Three-dimensional graph of the relationships between wavelengths 462 (x axis), 646 (y axis), 914 (z axis) and
        redshift (colour map)
        :return: plt object
        """
        xdata = self.data["UjMAG"]
        ydata = self.data["Mcz"]
        zdata = self.data["VjMAG"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(xs=xdata, ys=ydata, zs=zdata)

        fig.colorbar(img)
        ax.set_xlabel('UjMAG')
        ax.set_ylabel('Mcz')
        ax.set_zlabel('VjMAG')

        plt.show()

        return plt

    def three_band_luminosity(self):
        """
                Three-dimensional graph of the relationships between wavelengths 462 (x axis), 646 (y axis), 914 (z axis) and
                redshift (colour map)
                :return: plt object
                """
        xdata = self.data["UjMAG"]
        ydata = self.data["BjMAG"]
        zdata = self.data["VjMAG"]
        cdata = self.data["Mcz"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(xs=xdata, ys=ydata, zs=zdata, c=cdata, cmap=plt.hot())

        fig.colorbar(img)
        ax.set_xlabel("UjMAG")
        ax.set_ylabel("BjMAG")
        ax.set_zlabel("VjMAG")

        plt.show()

        return plt

    def three_band_flux(self):
        """
        Three-dimensional graph of the relationships between wavelengths 462 (x axis), 646 (y axis), 914 (z axis) and
        redshift (colour map)
        :return: plt object
        """
        xdata = self.data["W420FE"]
        ydata = self.data["W646FD"]
        zdata = self.data["W914FD"]
        cdata = self.data["Mcz"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(xs=xdata, ys=ydata, zs=zdata, c=cdata, cmap=plt.hot())

        fig.colorbar(img)
        ax.set_xlabel('Wavelength = 420nm')
        ax.set_ylabel('Wavelength = 646nm')
        ax.set_zlabel('Wavelength = 914nm')

        plt.show()

        return plt

    def two_axis_luminosity(self):
        data = pd.read_csv("data_sets/COMBO17.csv")
        """
        Two-dimensional graph of the relationship between galaxy luminosity (y axis) and redshift (x axis)
        :return: plt object
        """

        xdata = data["Mcz"]
        ydata = data["mumax"]

        plt.scatter(xdata, ydata, color="0.4", s=1)

        plt.title("Luminoisty VS Redshift for COMBO17 Galaxies")
        plt.xlabel("Redshift (Mcz)")
        plt.ylabel("Luminosity (μmax)")

        plt.show()

        return plt


if __name__ == "__main__":
    Graphical().two_axis_luminosity()
