import sys

# exception traceback for debugging
import traceback

from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import itertools

# shading
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import LightSource

# automatic gaussian filters for 2D (mgrids) and 1D (evenly-spaced-curves) arrays
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.feature import peak_local_max

# finding local maxima in 1D
from scipy.signal import find_peaks

from shapely.geometry import Polygon, Point, MultiPoint

# for watershed segmentation utilized in edge detection & correction
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours

# from scipy.ndimage import rotate

from scipy.optimize import curve_fit

# for saving and loading LDS object
import pickle



class LidarDataSet:
    """
    class for lidar data

    :param ndarray dataset: 2d array of height values...
    :param pandas.DataFrame sinkholes: array of sinkholes

    :param ndarray dataset_filtered:
    :param float maxR: maximum expected value of sinkhole radius.
                       It helps determine the analyzed region for finding sinkhole edges and radial distribution
                       It can be provided as argument of the __init__ method
                       , defaults to 250(m)
    :param float lenX: width of dataset
    :param float lenY: height of dataset
    :param Polygon analyzed_region: region to which we restrict 

    """

       

    def __init__(self, dataset, maxR=250, selected_range_points=None):
        """
        initialization method, it requires user to provide a ready spatial dataset in 2d ndarray format.
        To construct LDS instance from one ore multiple csv files, one can use provided classmethods.

        :param 2d numpy array dataset: array of height values
        :param int maxR: ceiling(max sinkhole radius value)
        :param list of lists of len 2 selected_range_poinst: list of points defining the edge of area we want to anlyze, defaults to None
                                                    
        """
        self.dataset = dataset
        self.dataset_filtered = None
        self.sinkholes = None

        self.maxR = maxR
        self.lenX, self.lenY = self.dataset.shape

        maxR, lenX, lenY = self.maxR, self.lenX, self.lenY # aliases
        dataset_region = Polygon([[0,0],[0,lenY],[lenX,lenY],[lenX,0]])
        nonborder_region = Polygon([[maxR,maxR],[maxR,lenY-maxR],[lenX-maxR,lenY-maxR],[lenX-maxR,maxR]])

        if selected_range_points is None:
            self.analyzed_region = dataset_region
            self.analyzed_nonborder_region = nonborder_region
        else:
            selected_range = Polygon(selected_range_points)
            self.analyzed_region = selected_range.intersection(dataset_region)
            self.analyzed_nonborder_region = selected_range.intersection(nonborder_region)

        buffer = self.analyzed_region.exterior.buffer(-maxR, single_sided=True)
        # TODO: check!!!!
        self.inner_region = self.analyzed_region.difference(buffer)

        # all_sinkholes = self.sinkholes["center"]
        # inner_sinkholes = [p for p in all_sinkholes if Point(p).within(buffer)]
        
    #=================================================================================================================== static methods
    @staticmethod
    def __data_tile_from_csv(filepath):
        """
        create 2d ndarray with height data from standard-formatted csv lidar data

        :param string filepath: path of the standard-format lidar data csv file

        :return: the dataset
        :rtype: ndarray

        """
        return pd.read_csv(filepath, delimiter=";", header=None).values[:,2].reshape(1000,1000)

    #=================================================================================================================== custom constructor methods
    @classmethod
    def from_csv_file(cls, filepath, maxR=250, selected_range_points=None):
        """
        create LidarDataSet object from single lidar data csv file
        we get 1 tile of size 1km x 1km

        :param string filename
        
        :return: the new instance of LDS
        :rtype: LidarDataSet

        """
        return cls(cls.__data_tile_from_csv(filepath), maxR=maxR, selected_range_points=selected_range_points)

    @classmethod
    def from_array_of_csv_files(cls, array_of_filenames, filename_prefix="", filename_extension=".txt", maxR=250, selected_range_points=None):
        """
        create LidarDataSet object from array of lidar data csv files
        
        :param (2d iterable of strings) array_of_filenames:
        :param string or None filename_prefix:
        :param string filename_extension:
        
        :return: the new instance of LDS
        :rtype: LidarDataSet

        """
        data_segments = [[cls.__data_tile_from_csv(f"{filename_prefix}{filename}{filename_extension}") for filename in row] for row in array_of_filenames]
        # dataset = np.concatenate([np.concatenate(row, axis=1) for row in data_segments], axis=0)
        dataset = np.vstack([np.hstack(row) for row in data_segments])
        return cls(dataset, maxR=maxR, selected_range_points=selected_range_points)

    @classmethod
    def from_npy_file(cls, filepath, maxR=250, selected_range_points=None):
        """
        create LidarDataSet instance from dataset saved previously in binary .npy file

        :param string filepath:

        :return: the new instance of LDS
        :rtype: LidarDataSet

        """
        with open(filepath, "rb") as f:
            dataset = np.load(f)
        return cls(dataset, maxR=maxR, selected_range_points=selected_range_points)

    @classmethod
    def from_rectangular_subset(cls, elds, min_x, max_x, min_y, max_y, maxR=250, selected_range_points=None):
        """
        creates new LDS instance from existing LDS (elds) by finding its subset
        
        :param LidarDataSet elds: LDS instance from which we extract the subset
        :param int min_x: lowest x coord value for new LDS
        :param int max_x: highest x coord value for new LDS
        :param int min_y: lowest y coord value for new LDS
        :param int max_y: highest y coord value for new LDS

        :return: the new instance of LDS
        :rtype: LidarDataSet

        """
        dataset = elds.dataset[min_x:max_x,min_y:max_y]
        nlds = cls(dataset, maxR=maxR, selected_range_points=selected_range_points)
        # TODO: copy also the existing sinkholes with all their attributes
        return nlds

    #=================================================================================================================== project memory
    # set of methods for saving and loading components of LidarDataSet instance
    # loading from and saving to csv files is less efficient and unnecessary if we're repeatedly working with the same dataset

    def save_dataset_to_npy(self, filepath):
        """
        save a copy of self.dataset to a given filepath
        storage format is numpy's binary .npy
        
        :param filepath: a path where .npy file is to be stored
        :type  filepath: string
        
        """
        with open(filepath,"wb") as f:
            np.save(f, self.dataset)

    def save_sinkholes_to_pkl(self, filepath):
        """
        save self.sinkholes to pandas pickle

        :param filepath:
        :type  filepath: string

        """
        self.sinkholes.to_pickle(filepath)

    def load_sinkholes_from_pkl(self, filepath):
        """
        load self.sinkholes from previously saved  pandas pickle file

        :param filepath:
        :type  filepath: string
        """
        self.sinkholes = pd.read_pickle(filepath)

    #=================================================================================================================== exporting data

    def get_sinkhole_positions(self):
        return self.sinkholes[["center_x", "center_y"]]


    #===================================================================================================================


    def find_sinkholes(self, filter_sigma=5, min_dist=4, verbose=True):
        """
        detect all sinkhole and store them in self.sinkholes \n
        function first filters the self.dataset using gaussian filter with parameter filter_sigma and assigns the result to self.dataset_filtered
        then function finds dataset's local minima which positions are sinkholes centers
        results are written to self.sinkholes with columns center_x, center_y and center=(center_x, center_y)
        function can be calibrated using parameter filter_sigma

        :param float filter_sigma: parameter for gaussian (low-pass) filter, defaults to 5
        :param float min_dist: parameter for minima-finding algorithm, defaults to 4 (suitable for experimental data)
        
        """
        self.dataset_filtered = gaussian_filter(self.dataset, sigma=filter_sigma)
        minima = np.array(peak_local_max(np.negative(self.dataset_filtered), min_distance = min_dist))

        if verbose:
            print(f"LOG: find_sinkholes: found {minima.shape[0]} sinkholes")

        self.sinkholes = pd.DataFrame(minima, columns=['center_x', 'center_y'])
        self.sinkholes["center"] = self.sinkholes.apply(lambda row: (row["center_x"], row["center_y"]), axis=1)

        # if self.analyzed_region is None:
        #     self.sinkholes["analyzed"] = False # I think it works? # TODO: check
        # else:
        self.sinkholes["analyzed"]           = self.sinkholes.apply(lambda row: True if Point(row["center"]).within(self.analyzed_region) else False, axis=1)
        self.sinkholes["analyzed_nonborder"] = self.sinkholes.apply(lambda row: True if Point(row["center"]).within(self.analyzed_nonborder_region) else False, axis=1)
        self.sinkholes["inner"]              = self.sinkholes.apply(lambda row: True if Point(row["center"]).within(self.inner_region) else False, axis=1)
        # not really a better solution :
        # self.sinkholes = self.sinkholes.assign(analyzed=lambda row: True if Point(row["center"]).within(self.analyzed_region) else False)
        
   


    def find_sinkhole_edges(self, verbose=True):
        """
        for each of the sinkholes in the analyzed region we find a polygon covering its area
        
        """
        maxR = self.maxR

        if self.sinkholes is None:
            # this also defines sefl.dataset_filtered
            self.find_sinkholes()

        # aliases
        sinkholes = self.sinkholes
        dataset_filtered = self.dataset_filtered

        # ---------------------------------------------------------------------- FIRST WE PREPARE WATERSHED SEGMENTATION
        # a 2D ndarray with 0s where no sinkole centers present and sinkhole index values in s. centers
        marker_mask = np.zeros((self.lenX, self.lenY))
        for idx, p in enumerate(sinkholes.center):
            marker_mask[p[0],p[1]] = idx + 1 # WE MUST USE NON-ZERO INDEXING BC ZERO MEANS NON-LABELED DATA
        # watershed segmentation (scipy) on filtered altitude map using sinkhole centers as flooding sources
        ws = watershed(np.array(dataset_filtered), markers=marker_mask)

        
        # we will iterate over all sinkholes and for each find a relevant r2_edge polygon and watershed segmentation region
        # and then define its area as the intersection of the two
        corr_edge_polygons = []
        for idx, row in sinkholes.iterrows():
            if row["analyzed_nonborder"]:
                if verbose:
                    print(f"finding edge #:{idx}")
                
                center_x = row["center_x"]
                center_y = row["center_y"]

                valAt = lambda p: dataset_filtered[p[0]][p[1]]  # more aesthetical and shorter lookup of mgrid values

                r_values = np.arange(maxR)
                theta_values = np.arange(0,2*np.pi,np.pi/25) # 50 values of theta for entire full angle

                # function transforming local polar coordinates (for specific minimum) to global matrix coordinates (not exactly cartesian w,k not x,y)
                polar_to_cart = lambda theta, r: (int(round(r*np.cos(theta)))+center_x, int(round(r*np.sin(theta)))+center_y)

                # a list of radial height functions for each theta
                polar_mesh = [[valAt(polar_to_cart(theta, r)) for r in r_values] for theta in theta_values]

                # for each radial f. in polar_mesh we calculate it's derivative function
                polar_mesh_ders = [np.gradient(func) for func in polar_mesh]

                # we gaussian filter all the derivatie funs
                polar_mesh_ders_filtered = [gaussian_filter1d(func, sigma=4) for func in polar_mesh_ders]

                theta_idx_values = []
                r2_values = []

                # finding r2 values
                for theta_idx in range(50):
                    # we analyze the filtered derivative polar function
                    fun = polar_mesh_ders_filtered[theta_idx]
                    maxima, _ = find_peaks(fun, height=0, distance=5)
                    # if a maximum was detected we proceed, else we ommit this theta
                    if maxima.size > 0:
                        r0 = maxima[0]
                        threshold = fun[r0] * 0.6
                        for r, y in enumerate(fun):
                            if r > r0 and y < threshold:
                                theta_idx_values.append(theta_idx)
                                r2_values.append(r)
                                break
                
                # case in which our detection has failed
                # TODO: we will improve this when we connect multiple tiles. For now we ommit this
                # and we assign circles of radius 48 as our r2 edge so that we can go further
                if len(r2_values) < 3:
                    r2_values = []
                    theta_idx_values = []
                    for theta_idx in range(50):
                        theta_idx_values.append(theta_idx)
                        r2_values.append(48)

                r2_values_filtered = gaussian_filter1d(r2_values, sigma=2, mode='wrap')
                theta_values_for_r0_values = [theta_values[theta_idx] for theta_idx in theta_idx_values]

                r2_edge_points = np.array([polar_to_cart(theta, r2) for theta, r2 in zip(theta_values_for_r0_values, r2_values_filtered)])
                r2_edge_polygon = Polygon(r2_edge_points)

                #=================================================================== WE PROCEED TO FINDING WS. SEGMENTATION REGION

                ws_idx = idx + 1 # watershed indexing is offset by 1 bc 0 denotes non-labeled data
                # we create a mask with i in the sinkhole area and 0 everywhere else
                ws_region = np.copy(ws) 
                ws_region[ws_region != ws_idx] = 0
                # skimage.measure function returning list of coordinate lists for each found contour
                # only shape present is the current watershedding region where value == i
                ws_contours = find_contours(ws_region) 
                # situation i don't think should ever occur, but if it does I want to know about it
                if len(ws_contours) != 1:
                    print("WARNING: multiple ws_contours!")
                    input()
                ws_polygon = Polygon(ws_contours[0])

                #=================================================================== WE CONNECT WATERSHEDDING REGIONS TO R2 POLYGONS
                try:
                    corr_edge_poly = r2_edge_polygon.intersection(ws_polygon)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    # fallback if failure
                    if verbose:
                        print(f"For sinkhole {idx} edge correction unsuccessful, fallback to original edges ")

                        print("Watershed")
                        print(type(ws_polygon))
                        print(ws_polygon.is_valid)
                        plt.imshow(ws_region.transpose(), origin="lower")
                        edge_polygon = r2_edge_polygon.exterior.xy
                        plt.plot(edge_polygon[0], edge_polygon[1], color="red")
                        plt.show()
                        input("input")
                    corr_edge_poly = r2_edge_polygon
                    
                # if corr_edge_poly is iterable (multipolygon e.g.) we chose one with the largest area
                try:
                    corr_edge_poly = corr_edge_poly[np.argmax([seg.area for seg in corr_edge_poly])]
                except TypeError:
                    # non-iterable case (single poly) => we don't have to do anything
                    pass
                
                corr_edge_polygons.append(corr_edge_poly)
            else:
                corr_edge_polygons.append(None)

        sinkholes["polygon"] = corr_edge_polygons



    #===================================================================================================== main routines




    #################################################################################################################### sinkhole statistics

 
    def get_sinkhole_sizes(self):
        """
        get sinkhole area and average radius values
        area is defined by shapely area function
        average radius is defined as sqrt(area/pi)

        :return: area values, average radius values
        :rtype: tuple(sequence<float>, sequence<float>)
        
        """
        if "area" not in self.sinkholes:
            self.sinkholes["area"] = self.sinkholes.apply(lambda row: row["polygon"].area, axis=1)
        if "avg_radius" not in self.sinkholes:
            self.sinkholes["avg_radius"] = self.sinkholes.apply(lambda row: np.sqrt(row["area"]/np.pi), axis=1)

        return self.sinkholes["area"], self.sinkholes["avg_radius"]


    def get_depth_values(self):
        """
        get depth values for all sinkholes for which we have found their edges

        :return: depth values
        :rtype: sequence<float>
        
        """
        if "depth" not in self.sinkholes:
            self.sinkholes["depth"] = self.sinkholes.apply(lambda row: np.average([self.dataset[round(p[0])][round(p[1])] for p in row["polygon"].exterior.coords]) - self.dataset[row["center_x"]][row["center_y"]], axis=1)
     
        return self.sinkholes["depth"]

    #=================================================================================================================== visualisation


    def plot(self,
             edges=False,
             centers=False,
             analyzed_region=False,
             hillshade=True,
             verbose=False,
             fig_size=None,
             save=False,
             filename="output.png"):
        """
        plot the dataset

        :param bool edges: plot sinkhole edges, defaults to False
        :param bool centers: plot sinkhole centers, defaults to False
        :param bool analyzed_region: mark the analyzed region of the dataset, defaults to False
        :param bool verbose: info on the progress, defaults to False
        :param iterable of size 2 or None fig_size: custom figure size, useful for jupyter notebook, where default is rather small, defaults to None
        :param bool save: whether to save to file, if not it's shown on screen, defaults to False
        :param string filename: filename (used if save=True)

        """
        print("PLOTTING")

        if centers or edges:
            sh    = self.sinkholes
            sh_an = sh[sh["analyzed"]]
            sh_annb = sh[sh["analyzed_nonborder"]]
            sh_in = sh[sh["inner"]]

        if fig_size:
            plt.figure(figsize=fig_size)

        # plot the map
        if hillshade:
            ls = LightSource(azdeg=135, altdeg=45)
            ve = 5  # 0.05
            plt.imshow(ls.shade(self.dataset.transpose(), blend_mode="hsv", vert_exag=ve, cmap=plt.cm.gist_earth), origin='lower')
        else:
            plt.imshow(self.dataset.transpose(), cmap=plt.cm.gist_earth, origin='lower')


        if centers:
            plt.scatter(sh["center_x"], sh["center_y"], color="darkblue", s=7.0)
            plt.scatter(sh_an["center_x"], sh_an["center_y"], color="red", s=3.0)
            plt.scatter(sh_annb["center_x"], sh_annb["center_y"], color="orange", s=3.0)
            plt.scatter(sh_in["center_x"], sh_in["center_y"], color="yellow", s=1.5)

        # plot the edge of the analyzed range
        if analyzed_region:
            analyzed_region_edge = self.analyzed_region.exterior.xy
            analyzed_nonborder_region_edge = self.analyzed_nonborder_region.exterior.xy
            inner_region_edge = self.inner_region.exterior.xy
            plt.plot(analyzed_region_edge[0],analyzed_region_edge[1], color="red")
            plt.plot(analyzed_nonborder_region_edge[0],analyzed_nonborder_region_edge[1], color="orange")
            plt.plot(inner_region_edge[0],inner_region_edge[1], color="yellow")

        # plot sinkhole edges
        if edges:
            if verbose:
                print("PLOTTING EDGES")
            for i, row in sh_annb.iterrows():
                if verbose:
                    print(f"plotting edge {i}")
                try:
                    edge_polygon = row["polygon"].exterior.xy
                    plt.plot(edge_polygon[0], edge_polygon[1], color="red")
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    print("ERROR: Plotting: exception occured")
                    traceback.print_exc()
                    # input()

        if save:
            plt.savefig(filename)
        else:
            plt.show()





  









