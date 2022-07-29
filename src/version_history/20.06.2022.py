# from astroquery.mast import Observations, Catalogs, Tesscut, Zcut
# from astropy.coordinates import SkyCoord
import datetime
# TODO:
#   a

from datetime import datetime

import requests

import xlsxwriter
import csv
import pickle
import json

import numpy as np
from os.path import exists

import pandas as pd
import scipy.stats as scipy
import impyute as impy
import numpy as np
import matplotlib.pyplot as plt
from conversions import *
from astropy.io import fits

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, OPTICS
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib import gridspec

from dataclasses import dataclass


class pkl(object):
    @staticmethod
    def save_model(model, name):
        with open(f'{name}_{datetime.now().date()}', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_model(name):
        with open(f'{name}', 'rb') as handle:
            model = pickle.load(handle)
        return model


class metrics(object):
    def __init__(self, model, class_instance=None):
        self.model = model
        self.cls_instance = class_instance

    def silhouette(self, generate_graph: bool = True, data=None):
        """
        This method creates a visualiser plot based upon a cluster model.
        :return: visualiser object
        """
        if type(self.model) != KMeans:
            raise TypeError(
                f"Wasn't able to evaluate silhouette metric from the model type: {type(self.model)}. "
                f"Requires KMeans model instead."
            )

        if self.cls_instance is not None:
            x_data = self.cls_instance.__getattribute__("X_train")
        else:
            x_data = data

        score = silhouette_score(X=x_data, labels=self.model.labels_)

        if generate_graph:
            visualiser = SilhouetteVisualizer(self.model, colors='yellowbrick')
            visualiser.fit(X=x_data)  # x_data is the combo_data array used in the creation of cluster model
            visualiser.show()
            return [visualiser, score]
        else:
            return score

    @staticmethod
    def inertia_graph(n_iterations: int):
        """
        This graphs the inertia score of the range of clusters between 2 and n_iterations.
        :param n_iterations: Number of iterations (k) calculated. The inertia of the model calculated by the
        number of clusters range (2, n_iterations) will be graphed.
        :return: plt object
        """
        _inertia_score = []
        k = range(2, n_iterations)
        for n in k:  # TODO: Make sure this works
            _inertia_score.append(Cluster().kmeans(n=n, save=False).inertia_)

        plt.plot(k, _inertia_score)
        plt.xlabel("n clusters")
        plt.ylabel("inertia")
        plt.show()

    def silhouette_graph(self, n_iterations: int):
        if type(self.model) != KMeans:
            raise TypeError(
                f"Wasn't able to evaluate inertia metric from the model type: {type(self.model)}. "
                f"Requires KMeans model instead."
            )
        _silhouette_score = []
        k = range(2, n_iterations)
        for n in k:  # TODO: Make sure this works
            cluster = Cluster()
            model = cluster.kmeans(n=n, save=False)
            _silhouette_score.append(metrics(class_instance=cluster, model=model).silhouette(generate_graph=False)[1])

        plt.plot(k, _silhouette_score)
        plt.xlabel("n clusters")
        plt.ylabel("inertia")
        plt.show()


@dataclass
class _Data:
class _COMBO_Data(object):
    """
    This class (_Data) constructs the arrays for various compilations of data
    This class (_Data) constructs the arrays for various compilations of combo_data
    The flux_to_luminosity class in conversions.py controls the conversion from flux to luminosity.
    Currently there is no conversion from flux to luminosity. I.e. flux_to_luminosity returns original flux parameter.
    """
    data = pd.read_csv("data_sets/COMBO17.csv")
    Incl_Mcz: np.ndarray = np.array(
        list(zip(data["Mcz"], data["UjMAG"], data["BjMAG"], data["VjMAG"]
                 )))
    Excl_Mcz: np.ndarray = np.array(
        list(zip(data["W420FE"], data["W462FE"], data["W485FD"],
                 data["W518FE"], data["W571FS"], data["W604FE"], data["W646FD"],
                 data["W696FE"], data["W753FE"], data["W815FS"], data["W856FD"],
                 data["W914FD"]
                 )))
    restricted_Incl_Mcz: np.ndarray = np.array(
        list(zip(data["Mcz"], data["UFS"], data["BFS"], data["VFD"],
                 data["Mcz"], data["RFS"], data["IFD"]
                 )))
    wtf_Incl_Mcz: np.ndarray = np.array(
        list(zip(
            data["Mcz"], data["W420FE"], data["W462FE"],
            data["W485FD"], data["W518FE"], data["W571FS"], data["W604FE"],
            data["W646FD"], data["W696FE"], data["W753FE"], data["W815FS"],
            data["W856FD"], data["W914FD"], data["UFS"], data["BFS"], data["VFD"],
            data["Mcz"], data["RFS"], data["IFD"]
        )))
    Intensity_Mag: np.ndarray = np.array(
        list(zip(data["Mcz"], data["mumax"]
                 )))
    if exists("combo_data"):
        """
        IF THE CLEANSED DATA HAS BEEN SAVED TO A FILE, IT WILL BE LOADED INSTEAD OF CLEANSING IT AGAIN
        """
        with open(f'combo_data', 'rb') as handle:
            save_data_output = pickle.load(handle)

        columns: np.ndarray = save_data_output[0]["columns"]
        Mcz: np.ndarray = save_data_output[0]["Mcz"]
        Incl_Mcz_Johnson: np.ndarray = save_data_output[0]["Incl_Mcz_Johnson"]
        Excl_Mcz_Johnson: np.ndarray = save_data_output[0]["Excl_Mcz_Johnson"]
        Incl_Mcz_Sloan: np.ndarray = save_data_output[0]["Incl_Mcz_Sloan"]
        Excl_Mcz_Sloan: np.ndarray = save_data_output[0]["Excl_Mcz_Sloan"]
        Mcz_prop: np.ndarray = save_data_output[0]["Mcz_prop"]
        percent_nan = save_data_output[1]

        print(percent_nan)

    else:
        with fits.open('data_sets/combo17.fits') as hdul:
            hdul.verify('fix')
            i_data = hdul[1].data
            cols = hdul[1].columns

        columns: fits.column.ColDefs = cols

        """
        CREATING A DATA MASK FOR PHOT_FLAG >= 8 AND STELLARITY < 0.25. 
        THIS FILTERS THE DATA TO REMOVE PROBLEMATIC DATAPOINTS AND ONLY INCLUDE STARS.
        """

        phot_mask = (i_data["phot_flag"] < 8)  # A mask created of the data to exclude erroneous data-points.
        galaxy_mask = (i_data["stellarity"] < 0.25)  # A mask created includeing all galaxies.

        total_mask = np.logical_and(phot_mask, galaxy_mask)  # Combining both phot and galaxy mask with an and statement

        data = i_data[total_mask]  # Application of the phot_flag mask to initial combo_data array.

        percent_nan = {  # Calculation for the percentage of missing combo_data in each of the used arrays.
            "MC_z": np.count_nonzero(np.isnan(data["MC_z"])) / data["MC_z"].size,

            "UjMAG": np.count_nonzero(np.isnan(data["UjMAG"])) / data["UjMAG"].size,
            "BjMAG": np.count_nonzero(np.isnan(data["BjMAG"])) / data["BjMAG"].size,
            "VjMAG": np.count_nonzero(np.isnan(data["VjMAG"])) / data["VjMAG"].size,

            "usMAG": np.count_nonzero(np.isnan(data["usMAG"])) / data["usMAG"].size,
            "gsMAG": np.count_nonzero(np.isnan(data["gsMAG"])) / data["gsMAG"].size,
            "rsMAG": np.count_nonzero(np.isnan(data["rsMAG"])) / data["rsMAG"].size,

            "x": np.count_nonzero(np.isnan(data["x"])) / data["x"].size,
            "y": np.count_nonzero(np.isnan(data["y"])) / data["y"].size,

            "MinAxis": np.count_nonzero(np.isnan(data["MinAxis"])) / data["MinAxis"].size,
            "MajAxis": np.count_nonzero(np.isnan(data["MajAxis"])) / data["MajAxis"].size,

            "dl": np.count_nonzero(np.isnan(data["dl"])) / data["dl"].size,
            "ApD_Rmag": np.count_nonzero(np.isnan(data["ApD_Rmag"])) / data["ApD_Rmag"].size,
            "mu_max": np.count_nonzero(np.isnan(data["mu_max"])) / data["mu_max"].size,
            "Rmag": np.count_nonzero(np.isnan(data["Rmag"])) / data["Rmag"].size,
        }

        """
        FORMATION OF DAUGHTER ARRAYS USED IN THE REGRESSION, CLUSTER AND DENSITY ANALYSIS
        """

        uncertainty = {
            "e_UjMag": data["e_UjMag"].mean(),
            "e_BjMag": data["e_BjMag"].mean(),
            "e_VjMag": data["e_VjMag"].mean(),
            "e_usMag": data["e_usMag"].mean(),
            "e_gsMag": data["e_gsMag"].mean(),
            "e_rsMag": data["e_rsMag"].mean()
        }

        print(uncertainty)

        Incl_Mcz_Johnson: np.ndarray = impy.em(np.array(
            list(zip(
                data["MC_z"], data["UjMAG"], data["BjMAG"], data["VjMAG"]
            )), dtype=np.float64
        ))

        Excl_Mcz_Johnson: np.ndarray = impy.em(np.array(
            list(zip(
                data["UjMAG"], data["BjMAG"], data["VjMAG"]
            )), dtype=np.float64
        ))

        Incl_Mcz_Sloan: np.ndarray = impy.em(np.array(
            list(zip(
                data["MC_z"], data["usMAG"], data["gsMAG"], data["rsMAG"]
            )), dtype=np.float64
        ))

        Excl_Mcz_Sloan: np.ndarray = impy.em(np.array(
            list(zip(
                data["usMAG"], data["gsMAG"], data["rsMAG"]
            )), dtype=np.float64
        ))

        Mcz_prop: np.ndarray = impy.em(np.array(
            list(zip(
                data["MC_z"],
                data["x"], data["y"],
                data["ApD_Rmag"],
                data["MajAxis"], data["MinAxis"],
                data["dl"],
                data["mu_max"],
                data["Rmag"]
            )), dtype=np.float64
        ))

        Mcz: np.ndarray = np.array(
            Incl_Mcz_Johnson[0]
        )

        """
        SAVING THE DATA TO A FILE SO THIS CLEANSING WILL NOT HAVE TO BE EVALUATED EACH TIME
        """

        save_data_input = [{
            "columns": columns,
            "Mcz": Mcz,
            "Incl_Mcz_Johnson": Incl_Mcz_Johnson,
            "Incl_Mcz_Sloan": Incl_Mcz_Sloan,
            "Excl_Mcz_Johnson": Excl_Mcz_Johnson,
            "Excl_Mcz_Sloan": Excl_Mcz_Sloan,
            "Mcz_prop": Mcz_prop
        }, percent_nan]

        with open(f'combo_data', 'wb') as handle:
            pickle.dump(save_data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Environment(_COMBO_Data):
    def __init__(self, *redshift_range):
        """

        :param redshift_range:
        """
        _COMBO_Data.__init__(self)

        self.a_min_z = redshift_range[0][0]
        self.a_max_z = redshift_range[0][1]
        self.b_min_z = None
        self.b_max_z = None

    def density_graph(self, bins: int = 0.002):
        """
        Two-dimensional graph of the relationship between galaxy luminosity (y axis) and redshift (x axis)
        :return: plt object
        """

        array = np.arange(0.1, 1.91, 0.01)
        values = [[], []]
        for i in array:
            _min, _max = i-bins, i+bins

            a_mcz_mask = np.logical_and((_min < self.Mcz_prop[:, 0]), (_max > self.Mcz_prop[:, 0]))
            a_Mcz_prop = self.Mcz_prop[a_mcz_mask]
            a_galaxies = [a_Mcz_prop[:, 1], a_Mcz_prop[:, 2]]

            if len(a_galaxies[0])==0:
                a_galaxies[0]=[0]
            if len(a_galaxies[1])==0:
                a_galaxies[1]=[0]

            a_x_range = min(a_galaxies[0]) - max(a_galaxies[0])
            a_y_range = min(a_galaxies[1]) - max(a_galaxies[1])

            values[0].append(i)
            if not a_galaxies[0][0]==0:
                values[1].append(len(a_galaxies[0]) / (a_x_range * a_y_range))
            else:
                values[1].append(0)

        redshift_mask = (self.Incl_Mcz_Johnson[:, 0] < 2)

        x_values1 = np.array(values[0])
        y_values1 = np.array(values[1])

        x_values2_r = self.Incl_Mcz_Johnson[:, 0][redshift_mask]
        y_values2_u = self.Incl_Mcz_Johnson[:, 1][redshift_mask]
        y_values2_v = self.Incl_Mcz_Johnson[:, 2][redshift_mask]
        y_values2_b = self.Incl_Mcz_Johnson[:, 3][redshift_mask]

        fig, ax1 = plt.subplots(figsize=(9, 6))

        ax2 = ax1.twinx()

        ax1_colour = "0.4"
        ax2_colour = "0"

        ax1.scatter(x_values2_r, y_values2_u+0, color=ax1_colour, s=0.1)
        ax1.scatter(x_values2_r, y_values2_b+10, color=ax1_colour, s=0.1)
        ax1.scatter(x_values2_r, y_values2_v+20, color=ax1_colour, s=0.1)
        ax2.plot(x_values1, y_values1, color=ax2_colour)

        ax1.set_xlabel("Redshift (z)")
        ax1.set_ylabel("Luminosity (mag)", color=ax1_colour)
        ax1.tick_params(axis="y", labelcolor=ax1_colour)

        ax2.set_ylabel("Density (n galaxies/pixel^2)", color=ax2_colour)
        ax2.tick_params(axis="y", labelcolor=ax2_colour)

        fig.suptitle("Overlay of Luminosity VS Redshift (Left) and Density VS Redshift (Right) in COMBO17 Survey", fontsize=12)
        ax1.grid(b=None)
        ax2.grid(b=None)
        plt.show()

    def density_histogram(self, bins: int = 50):
        """
        Generates a histogram fo the density of galaxies across the cfds_r.fit image
        :param bins: [1, 2] array
        :return: density of galaxies per pixel^2
        """

        """
        DENSITY
        """

        a_mcz_mask = np.logical_and((self.a_min_z < self.Mcz_prop[:, 0]), (self.a_max_z > self.Mcz_prop[:, 0]))
        a_Mcz_prop = self.Mcz_prop[a_mcz_mask]
        a_galaxies = [a_Mcz_prop[:, 1], a_Mcz_prop[:, 2]]

        a_x_range = min(a_galaxies[0]) - max(a_galaxies[0])
        a_y_range = min(a_galaxies[1]) - max(a_galaxies[1])

        a_density = len(a_galaxies[0]) / (a_x_range * a_y_range)

        """
        HISTOGRAM
        """

        a_mcz_mask = np.logical_and((self.a_min_z < self.Mcz_prop[:, 0]), (self.a_max_z > self.Mcz_prop[:, 0]))
        a_Mcz_prop = self.Mcz_prop[a_mcz_mask]
        a_galaxies = [a_Mcz_prop[:, 1], a_Mcz_prop[:, 2]]

        plt.hist2d(a_galaxies[0], a_galaxies[1], bins=(bins, bins), cmap=plt.cm.Greys)
        plt.colorbar(label='n Galaxies')

        # Add labels
        plt.title(f'Density of galaxies in cdfs_r.fit image ({self.a_min_z}<z<{self.a_max_z})')
        plt.xlabel('x coordinate (pixels)')
        plt.ylabel('y coordinate (pixels)')

        plt.show()

class Regression(_Data):
    def __init__(self):
        _Data.__init__(self)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Excl_Mcz,
            self.data["Mcz"],
            test_size=0.2,
            random_state=None
        return a_density

    def density(self):
        """
        Calculates the density of a galaxy in a determined redshift range
        :return:
        """

        a_mcz_mask = np.logical_and((self.a_min_z < self.Mcz_prop[:, 0]), (self.a_max_z > self.Mcz_prop[:, 0]))
        a_Mcz_prop = self.Mcz_prop[a_mcz_mask]
        a_galaxies = [a_Mcz_prop[:, 1], a_Mcz_prop[:, 2]]

        a_galaxies_array = np.array(a_galaxies)
        a_mean = a_galaxies_array.mean(axis=1)
        a_var = a_galaxies_array.var(axis=1)

        a_observations = [len(a_galaxies[0]), len(a_galaxies[1])]

        a_x_range = min(a_galaxies[0]) - max(a_galaxies[0])
        a_y_range = min(a_galaxies[1]) - max(a_galaxies[1])

        a_density = len(a_galaxies[0]) / (a_x_range * a_y_range)

        return {"observations": a_observations, "mean": a_mean, "var": a_var, "density": a_density}

    def average_size(self):
        """
        No statistical significance in the difference of mean galaxy sizes when analysing through unpaired t-test
            regardless of assumption regarding equal variances.
        :return: {"mean": mean_average, "stv": standard_dev, "t_test": t_stat}
                 or {"mean": mean_average, "stv": standard_dev}
        """

        a_values = []
        for i in self.Mcz_prop:
            if self.a_min_z < i[0] < self.a_max_z:
                a_values.append(abs(i[3]))
        mean_average = np.mean(a_values)
        var = np.var(a_values)
        observations = [len(a_values[0]), len(a_values[1])]
        df = [len(a_values[0]) - 1, len(a_values[1]) - 1]

        if self.b_min_z is not None and self.b_max_z is not None:
            b_values = []
            for i in self.Mcz_prop:
                if self.b_min_z < i[0] < self.b_max_z:
                    b_values.append(abs(i[3]))
            t_stat = scipy.ttest_ind(a_values, b_values, equal_var=True)

            return {"observations": observations, "df": df, "mean": mean_average, "var": var, "t_test": t_stat}
        else:
            return {"observations": observations, "df": df, "mean": mean_average, "var": var}

    def average_maj_min_axis(self, equal_var=False):
        """

        :param equal_var:
        :return:
        """
        a_minaxis_values = []
        a_majaxis_values = []
        for i in self.Mcz_prop:
            if self.a_min_z < i[0] < self.a_max_z:
                a_minaxis_values.append(i[4])
        for i in self.Mcz_prop:
            if self.a_min_z < i[0] < self.a_max_z:
                a_majaxis_values.append(i[5])
        minaxis_mean_average = np.mean(a_minaxis_values)
        minaxis_standard_dev = np.std(a_minaxis_values)
        majaxis_mean_average = np.mean(a_majaxis_values)
        majaxis_standard_dev = np.std(a_majaxis_values)

        if self.b_min_z is not None and self.b_max_z is not None:
            b_minaxis_values = []
            b_majaxis_values = []
            for i in self.Mcz_prop:
                if self.b_min_z < i[0] < self.b_max_z:
                    b_minaxis_values.append(i[4])
            for i in self.Mcz_prop:
                if self.b_min_z < i[0] < self.b_max_z:
                    b_majaxis_values.append(i[5])
            minaxis_t_stat = scipy.ttest_ind(a_minaxis_values, b_minaxis_values, alternative="less")
            majaxis_t_stat = scipy.ttest_ind(a_majaxis_values, b_majaxis_values, alternative="less")

            return {
                "min_axis": {"mean": minaxis_mean_average, "stv": minaxis_standard_dev, "t_test": minaxis_t_stat},
                "maj_axis": {"mean": majaxis_mean_average, "stv": majaxis_standard_dev, "t_test": majaxis_t_stat}
            }

        else:
            return {
                "min_axis": {"mean": minaxis_mean_average, "stv": minaxis_standard_dev},
                "maj_axis": {"mean": majaxis_mean_average, "stv": majaxis_standard_dev}
            }

    def average_DL(self):
        a_values = []
        for i in self.Mcz_prop:
            if self.a_min_z < i[0] < self.a_max_z:
                a_values.append(abs(i[6]))
        mean_average = np.mean(a_values)
        standard_dev = np.std(a_values)

        if self.b_min_z is not None and self.b_max_z is not None:
            b_values = []
            for i in self.Mcz_prop:
                if self.b_min_z < i[0] < self.b_max_z:
                    b_values.append(abs(i[6]))
            t_stat = scipy.ttest_ind(a_values, b_values, equal_var=True)

            return {"mean": mean_average, "stv": standard_dev, "t_test": t_stat}
        else:
            return {"mean": mean_average, "stv": standard_dev}


class Regression(_COMBO_Data):
    def __init__(self, test_size, random_state=None):
        """
        This contains all regression analysis applied to the data. This initialise method creates all required arrays
        (test and training) that will be required for the regression analysis.
        :param test_size: The percentage of the parent array that the test array is taken from
        :param random_state: The random seed given to the regression model
        """
        _COMBO_Data.__init__(self)
        self.X_train, self.X_test = train_test_split(
            self.Excl_Mcz_Johnson,
            test_size=test_size,
            random_state=random_state
        )
        self.y_train, self.y_test = train_test_split(
            self.Mcz,
            test_size=test_size,
            random_state=random_state
        )

    def polynomial(self, degree: int, include_bias=False):
    def polynomial(self, degree: int, save: bool = True, include_bias: bool = False):
        """
        Constructs polynomial regression line with degree: degree. No significant relationships drawn from this.
        :param save: Boolean controls whether the model is saved to a file or not. (True = Save, False = Not Save)
        :param degree: Degree of the polynomial regression line
        :param include_bias: Dets the coefficients of certain variables to form a certain intercept.
        :return: type:list -> [regression model, regression score]
        """
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        _x_train = poly.fit_transform(self.X_train)
        _x_test = poly.fit_transform(self.X_test)
        reg = LinearRegression()
        reg.fit(_x_train, self.y_train)
        predicted = reg.predict(_x_test)
        score = r2_score(predicted, self.y_test)
        reg.fit(_x_train, y=self.y_train)
        score = reg.score(_x_test, self.y_test)
        if save:
            pkl.save_model(model=reg, name="polynomial_reg")
        return [reg, score]

    def linear(self):
    def linear(self, save: bool = True):
        """
        Runs a linear regression analysis. No significance given by score.
        :return: type:list -> [regression model, regression score]
        Runs a linear regression analysis.
        :param save: Boolean controls whether the model is saved to a file or not. (True = Save, False = Not Save)
        :return: Linear regression model
        """
        reg = LinearRegression().fit(self.X_train, self.y_train)
        score = reg.score(self.X_test, self.y_test)
        return [reg, score]
        if save:
            pkl.save_model(model=reg, name="linear_reg")
        return reg


class Cluster(_Data):
    def __init__(self, test_size, random_state):
        _Data.__init__(self)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Excl_Mcz,
            self.data["Mcz"],
            test_size=test_size,
            random_state=random_state
        )

    def cluster_Excl_Mcz(self, n):
class Cluster(_COMBO_Data):
    def __init__(self):
        """
        This method creates a cluster analysis excluding the redshift variable
            7-8 is the optimum grouping
        :param n: Number of CLuster Groups
        :return: None
        This contains all cluster analysis applied to the data. This initialise method creates all required arrays
        that will be required for the cluster modelling.
        """
        model = KMeans(n_clusters=n)
        model.fit(self.Excl_Mcz)
        silhouette = silhouette_score(self.Excl_Mcz, model.labels_)
        inertia = model.inertia_
        seed = model.random_state
        return [model, silhouette, inertia, seed]
        _COMBO_Data.__init__(self)
        self.X_train = self.Incl_Mcz_Johnson

    def cluster_Incl_Mcz(self, n, visualiser=False, save=True):
    def kmeans(self, n, save: bool = True):
        """
        This method creates a cluster analysis including the redshift variable
            3-4 is the optimum grouping
        :param save: Boolean controls whether the model is saved to a file or not. (True = Save, False = Not Save)
        :param n: Number of CLuster Groups
        :return: None
        :return: Cluster model
        """
        model = KMeans(n_clusters=n)
        model.fit(X=self.X_train, y=self.y_train)
        # y_predict = model.predict(X=self.X_test)
        # r2score = r2_score(y_true=self.y_test, y_pred=y_predict)
        silhouette = silhouette_score(X=self.X_train, labels=model.labels_)
        inertia = model.inertia_
        seed = model.random_state
        model.fit(X=self.X_train)
        # y_predict = model.predict(X=self.X_test)  # TODO: add method to evaluate accuracy of prediction
        if save:
            pickle.dump(model, open(f"model.pkl", "wb"))
        if visualiser:
            # TODO: Replace this with a more robust way of generating the silhouette visualiser
            visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
            visualiser.fit(X=self.X_train, y=self.y_train)  # _data is the data array used in the creation of cluster model
            visualiser.show()

        return [model, inertia, silhouette, seed]

    def visualiser(self, model, dataset=None):
        """
        This method creates a visualiser plot based upon a cluster model.
        :param dataset:
        :param Include_Mcz: True: Incl_Mcz, False: Excl_Mcz
        :param model: k-means cluster model
        :return: visualiser object
        """
        # TODO: Replace this with a more robust way of generating the silhouette visualiser
        _data = dataset
        visualiser = SilhouetteVisualizer(model, colors='yellowbrick')
        visualiser.fit(_data)  # _data is the data array used in the creation of cluster model

        visualiser.show()

        return visualiser


class cluster_operations(object):
    # TODO: Replace this with a more robust way of generating metrics
    def silhouette_graph(self, n_iterations: int):
        """
        This graphs the silhouette score of the range of clusters between 2 and n_iterations.
        :param n_iterations: Number of iterations (k) calculated. The silhouette score of the model calculated by the
        number of clusters range (2, n_iterations) will be graphed.
        :return: plt object
        """
        inertia_score = []
        k = range(2, n_iterations)
        for n in k:
            inertia_score.append(object[1])

        plt.plot(k, inertia_score)
        plt.xlabel("n clusters")
        plt.ylabel("silhouette score")
        plt.show()

        return plt

    def inertia_graph(self, n_iterations: int):
        """
        This graphs the inertia score of the range of clusters between 2 and n_iterations.
        :param n_iterations: Number of iterations (k) calculated. The inertia of the model calculated by the
        number of clusters range (2, n_iterations) will be graphed.
        :return: plt object
        """
        inertia_score = []
        k = range(2, n_iterations)
        for n in k:
            inertia_score.append(Cluster(random_state=None, test_size=0.2).cluster_Incl_Mcz(n)[2])

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
            pkl.save_model(model=model, name=f"kmeans_n-clusters={n}")

            figure, axis = plt.subplots(3, 2)
        return model

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

    def three_band_test(self):
        """
        Three-dimensional graph of the relationships between wavelengths 462 (x axis), 646 (y axis), 914 (z axis) and
        redshift (colour map)
        :return: plt object
        """
        xdata = self.data["W462FE"]
        ydata = self.data["Mcz"]
        zdata = self.data["W914FD"]

        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(xs=xdata, ys=ydata, zs=zdata)

        fig.colorbar(img)
        ax.set_xlabel('Wavelength = 462nm')
        ax.set_ylabel('Wavelength = Mcz')
        ax.set_zlabel('Wavelength = 914nm')

        plt.show()

        return plt

    def three_band(self):
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
class ManualLabour(object):
    @staticmethod
    def best_model(n_iterations, n_clusters):
    def best_model(n_iterations, n_clusters, save: bool = True):
        """
        Function to optimise the random_state for the cluster
        :param save:
        :param n_iterations: Number of initialisations of the cluster model
        :param n_clusters: Number of cluster groups in each iteration
        :return: optimum cluster model
        """
        best_model = [None, 0, 0]
        best_model = [None, None, 0]
        logs = {}

        for x in range(0, n_iterations):
            if Cluster().cluster_Incl_Mcz(n_clusters)[1] > best_model[1]:
                best_model = Cluster().cluster_Incl_Mcz(n_clusters)
                logs[x] = ["best", best_model[1], best_model[2]]
            initialised_class = Cluster()
            model = initialised_class.kmeans(n=n_clusters, save=False)
            current_score = metrics(class_instance=initialised_class, model=model).silhouette(generate_graph=False)
            if current_score > best_model[2]:
                best_model = [initialised_class, model, current_score]
                logs[x] = ["best", best_model[2]]
            else:
                logs[x] = "not best"
            print(x)

        return [best_model, logs]
        if save:
            pkl.save_model(model=best_model[1], name=f"kmeans_best_n-clusters-{n_clusters}_n-iterations-{n_iterations}")

        return best_model, logs

    @staticmethod
    def Excel_Export(array, Workbook_Name):
    def Excel_Export(array, workbook_name):
        """
        Exports an array to a excel workbook name given by the parameter Workbook_Name
        :param array: Numpy Array
        :param Workbook_Name: Name given to the excel workbook
        :param workbook_name: Name given to the excel workbook
        :return: None
        """
        workbook = xlsxwriter.Workbook(f'{Workbook_Name}.xlsx')
        workbook = xlsxwriter.Workbook(f'{workbook_name}.xlsx')
        worksheet = workbook.add_worksheet()

        row = 0

        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)

        workbook.close()

        return None


if __name__ == "__main__":
    cluster_operations().inertia_graph(n_iterations=20)
    # Cluster(random_state=None, test_size=0.2).cluster_Incl_Mcz(n=2, visualiser=False)
    Environment([0.459307, 0.499307], [0.981417, 1.021417]).density_graph()

