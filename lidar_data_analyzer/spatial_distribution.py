import itertools
import numpy as np
import matplotlib.pyplot as plt
from lidar_data_analyzer.LidarDataSet import LidarDataSet
from scipy.interpolate import CubicSpline #, PchipInterpolator, Akima1DInterpolator, BSpline, interp1d
from scipy.special import jv

# TODO: this function should perhaps be a method in LDS class
# we can write sth more general (taking just some random array instead of LDS instance)

def radial_distribution(lds: LidarDataSet,
                        method: str,
                        plot: bool = True,
                        spline: bool = False,
                        maxR: int = 250,
                        n_bins: int = 30
                        ) -> dict:
    """
    calculates the radial distribution function g(r) for provided LDS object,
    two methods possible
    1. "pbc" -> periodic boundary conditions
    2. "nonborder" -> radial distribution is calculated only for the inner sinkholes
                      using all the sinkholes within radius of maxR

    :param LidarDataSet lds: dataset for which to calculate the distribution
    :param str method: "pbc" or "nonborder"
    :param bool plot: whether to plot the obtained distribution, defaults to True

    :return: dictionary of radial distrib parameters
              "bin_edges",
              "bin_center_positions",
              "radial_distribution_values",
              "method"
    :rtype: dict
    
    """

    #--------------------------------------------------------------------------- validation

    if method != "pbc" and method != "nonborder":
        raise ValueError("wrong value for arg method, acceptable values: \"pbc\" or \"nonborder\"")

    #---------------------------------------------------------------------------
    lenX, lenY = lds.lenX, lds.lenY # de facto aliases for readibility
    
    # inner sinkholes are the ones for which we're actualy calculating radial distribution
    # for all shs in inner_shs we analyze surrounding of radius maxR
    # all_shs inclued inner_shs + buffer of width maxR

    ldss = lds.sinkholes # just shorter alias

    if method == "pbc":    
        sh = ldss[["center_x","center_y"]].to_numpy()
        N = len(sh)

    elif method == "nonborder":
        # positions of all inner sinkholes
        ish = ldss[ldss["inner"]][["center_x","center_y"]].to_numpy()
        ash = ldss[ldss["analyzed"]][["center_x","center_y"]].to_numpy()
        # positions of all buffer sinkholes (analyzed but not inner)
        # osh = ldss[(ldss["analyzed"]) & (ldss["inner"] == False)][["center_x","center_y"]].to_numpy()
        N = ish.shape[0]

    max_possible_maxR = min([lenX, lenY]) / 2
    maxR = min(int(max_possible_maxR), 250)

    area = lenX * lenY if method == "pbc" else lds.inner_region.area
    density = N / area

    distances = []

    if method == "nonborder":
        for shi in ish:
            for shj in ash:
                d = np.sqrt((shi[0]-shj[0])**2 + (shi[1]-shj[1])**2)
                if 1 < d <= maxR:
                    distances.append(d)

    elif method == "pbc":
        for i in range(N):
            for j in range(i+1, N):
                dx0 = abs(sh[i][0]-sh[j][0])
                dy0 = abs(sh[i][1]-sh[j][1])
                dx = min(dx0, abs(lenX - dx0))
                dy = min(dy0, abs(lenY - dy0))
                d = np.sqrt(dx ** 2 + dy ** 2)
                if 1 < d <= maxR:
                    distances.append(d)
            
    dr = maxR / n_bins
    bins, bin_edges = np.histogram(distances, bins=n_bins)
    r_vals = np.zeros(n_bins) # bin centers == average r values for bin
    rd = np.zeros(n_bins)
    
    if method == "nonborder":
        for i in range(n_bins):
            r_vals[i] = (bin_edges[i] + bin_edges[i+1]) / 2
            rd[i] = bins[i] / (2 * np.pi * N  * r_vals[i]  * dr * density)
    else: # pbc
        for i in range(n_bins):
            r_vals[i] = (bin_edges[i] + bin_edges[i+1]) / 2
            # in case of pbc we don't double count so...
            rd[i] = bins[i] / (np.pi * N  * r_vals[i]  * dr * density)

    if plot:
        if spline:
            plt.scatter(r_vals, rd, label="g(r)")
            cspline = CubicSpline(r_vals, rd)
            x_range = np.arange(r_vals[0], r_vals[-1], 1)
            plt.plot(x_range, cspline(x_range), label="g(r) spline")
        else:
            plt.plot(r_vals, rd, marker="o", linestyle="-", label="g(r)")
        plt.legend()
        plt.xlabel("r[m]")
        plt.show()

    if method == "pbc":
        bins *= 2

    return {
        "bin_edges": bin_edges,
        "bin_center_positions": r_vals,
        "radial_distribution_values": rd,
        "method": method,
        "N": N,
        "density": density,
        "raw_bins": bins
    }


def structure_factor_from_g_r(bin_edges: np.ndarray,
                              raw_bins: np.ndarray,
                              density: float,
                              N: int):
    """
    cacluate structure factor using data from radial distribution
    : param ndarray bin_edges: 1d ndarray of floats of length n
    : param ndarray g_r_values: 1d ndarray of floats of length (n - 1)

    : returns :
    : rtype :
    """
    # THIS WORKS ONLY IF YOU USE NUMPY ARGS
    # TODO: ADD VALIDATORS
    
    dr = bin_edges[1] - bin_edges[0]

    bin_r_min = bin_edges[:-1]
    bin_r_max = bin_edges[1:]
    bin_r_avg = (bin_r_min + bin_r_max) / 2

    bins_cumulative = raw_bins.cumsum()
    NR = 1 + bins_cumulative / N
    NR_avg = np.pi * density * (bin_r_max + dr/2)  ** 2
    SNR = NR - NR_avg
    S0R = SNR / (1 - (np.pi * density * bin_r_max ** 2 / N))


    # # Bessel function of the first kind, of order 0 (first arg)
    # # for r in all bins
    # # for k as in dft, so dk = 1 / (n * dr)
    # # and number of k args is n // 2 (possibly even less)
    # dk = 1 / (N * dr)

    # # jv can take array-like argument: jv(0, z_values)
    

    # for k in (i * dk for i in range(N // 2)):

    #     NkR[k] = NR * jv(0,k * bin_r_avg)
    #     # we have to multiply bins
    #     NkR[k] = [NR_bin * jv(0, k * r, out=None) for r, NR_bin in zip(bin_r_avg, NR)] # theoretically r values are supposed to be per point, but let's try it this way
 
    #     scipy.special.jv(0, k * r, out=None)


    fig, ax = plt.subplots(2)
    ax[0].plot(bin_r_max, NR, label='NR')
    ax[0].plot(bin_r_max, NR_avg, label='NR_avg')
    ax[0].legend()
    
    ax[1].plot(bin_r_max, SNR, label='SNR')
    ax[1].plot(bin_r_max, S0R, label='S0R')
    ax[1].legend()
    
    plt.show()

    return {
        "bin_r_min" : bin_r_min,
        "bin_r_max" : bin_r_max,
        "bin_r_avg" : bin_r_avg,
        "NR": NR,
        "NR_avg": NR_avg,
        "SNR": SNR,
        "S0R": S0R
    }

    # r_max_vals = bin_max_positions
    # L = len(r_max_vals)
    # # cumulative radial distribution
    # binsc = [sum(bins[0:i+1]) for i in range(len(rd))]
    
    # # <N(R)> WE DON'T MULTIPLY *2 BC IT'S ALREADY INCLUDED IN BINSC & BINS
    # NR = [1 + (binsc[i] / N ) for i in range(L)]
    # # we have to use r_max_vals (bc cumulative rd uses entire bins)
    # NR_avg = [ np.pi * density * r_max_vals[i]**2 for i in range(L)]
    # SNR = [(NR[i] - NR_avg[i]) for i in range(L)]
    # S0R = [SNR[i] / (1 - (np.pi * density * r_max_vals[i]**2 / N)) for i in range(L)]