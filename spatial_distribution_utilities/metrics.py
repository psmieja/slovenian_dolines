import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from numba import jit
from scipy.special import jv
from scipy.fft import fft, rfft, dst, fftfreq # for S(k)
from sklearn.neighbors import NearestNeighbors # used in psi_k

@jit(nopython=True)
def _calculate_radial_distribution(pa, pi, maxR, n_bins, density):
    """
    inner function for more efficient computation of radial distribution
    could be further optimized using some nearest neighbor finding algorithm
    also could be generalized for 3+ dimensionss
    """
    
    ni = pi.shape[0]
    na = pa.shape[0]
    dr = maxR / n_bins # hist bin width

    # distance matrix
    dm = np.zeros((ni, na))

    eps = 0.00001

    for i in range(ni):
        for a in range(na):
            d = np.sqrt((pi[i][0]-pa[a][0])**2 + (pi[i][1]-pa[a][1])**2)
            if eps < d <= maxR:
                dm[i,a] = d
    
    dv = dm.flatten()
    dv = dv[dv != 0]

    bins, bin_edges = np.histogram(dv, bins=[i*dr for i in range(n_bins + 1)])

    r_vals = np.zeros(n_bins) # bin centers == average r values for bin
    rd = np.zeros(n_bins)

    for i in range(n_bins):
        r_vals[i] = (bin_edges[i] + bin_edges[i+1]) / 2
        rd[i] = bins[i] / (2 * np.pi * ni  * r_vals[i]  * dr * density)
    
    return bins, bin_edges, r_vals, rd


def radial_disribution(points,
                       analyzed_region_pts,
                       maxR,
                       n_bins = 50,
                       plot=True,
                       plot_points=False):
    """
    calculates radial distribution  function for a defined

    :param (dataframe with columns x,y)            points: x,y positions of points for which the distribution is calculting
    :param (2d iterable)              analyzed_region_pts: x,y positions of points enclosing the area where the points lie
    :param float                                     maxR: max argument for the rdf and also the margin within the 'region' for the points from which rdf is not calculated
    :param int                                     n_bins: number of r values in rdf, defaults to 50
    :param bool                                      plot: whether to plot the rdf, defaults to True

    :returns :
    :rtype: dict
    """
    
    analyzed_region = Polygon(analyzed_region_pts)

    # using pandas is definately not most efficient but I'm in a hurry and I'm reusing code from LidarDataSet
    buffer = analyzed_region.exterior.buffer(-maxR, single_sided=True)
    inner_region = analyzed_region.difference(buffer)

    points["analyzed"] = points.apply(lambda row: True if Point([row["x"], row["y"]]).within(analyzed_region) else False, axis=1)
    points["inner"]    = points.apply(lambda row: True if Point([row["x"], row["y"]]).within(inner_region) else False, axis=1)

    points_analyzed = points[points["analyzed"]]
    points_inner = points[points["inner"]]

    pi = points_inner[["x","y"]].to_numpy()
    pa = points_analyzed[["x","y"]].to_numpy()

    density = len(points_analyzed) / analyzed_region.area

    print("Log: Calculating distances")

    bins, bin_edges, r_vals, rd = _calculate_radial_distribution(pa, pi, maxR, n_bins, density)

    if plot_points:

        analyzed_region_edge = analyzed_region.exterior.xy
        inner_region_edge = inner_region.exterior.xy

        fig, ax = plt.subplots()
        ax.scatter(points.x, points.y, s=2, color="blue")
        ax.scatter(points_analyzed.x, points_analyzed.y, s=2, color="red")
        ax.scatter(points_inner.x, points_inner.y, s=2, color="orange")
        ax.plot(analyzed_region_edge[0], analyzed_region_edge[1], color="red")
        ax.plot(inner_region_edge[0], inner_region_edge[1], color="orange")
        ax.set_aspect('equal','box')
        plt.show()

    if plot:
        plt.plot(r_vals, rd)
        plt.title("rdf")
        plt.show()
    
    return {"r": r_vals,
            "rd": rd,
            "bin_edges": np.array(bin_edges),
            "raw_bins": np.array(bins),
            "N_inner": len(points_inner),
            "N_analyzed": len(points_analyzed),
            "density": density
           }


def structure_factor_from_g_r_cich(bin_edges: np.ndarray,
                                   raw_bins: np.ndarray,
                                   density: float,
                                   N: int,
                                   plot = True):
    """
    cacluate structure factor using data from radial distribution
    : param ndarray bin_edges: 1d ndarray of floats of length n
    : param ndarray g_r_values: 1d ndarray of floats of length (n - 1)

    : returns :
    : rtype :
    """
    # THIS WORKS ONLY IF YOU USE NUMPY ARGS
    # TODO: ADD VALIDATORS
    
    print(type(bin_edges))

    dr = bin_edges[1] - bin_edges[0]

    bin_r_min = bin_edges[:-1]
    bin_r_max = bin_edges[1:]
    bin_r_avg = (bin_r_min + bin_r_max) / 2

    bins_cumulative = raw_bins.cumsum()
    NR = 1 + bins_cumulative / N
    NR_avg = np.pi * density * (bin_r_max)  ** 2 #np.pi * density * (bin_r_max + dr/2)  ** 2
    SNR = NR - NR_avg
    S0R = SNR / (1 - (np.pi * density * bin_r_max ** 2 / N))

    # Bessel function of the first kind, of order 0 (first arg)
    # for r in all bins
    # for k as in dft, so dk = 1 / (n * dr)
    # and number of k args is n // 2 (possibly even less)
    dk = 1 / (N * dr)

    # jv can take array-like argument: jv(0, z_values)
    nk = N // 2
    nr = bin_r_max.shape[0]
    NkR = np.empty((nk, nr))

    k = [i * dk for i in range(nk)]

    # for ir in range(nr):
    #     for ik in range(nk):
    #         NkR[ik,ir] = NR[ik,ir] * jv(0,k[ik] * bin_r_avg[ir], out=None)
    #         # we have to multiply bins
    #         NkR[ik,ir] = NR_bin * jv(0, k[ik] * r, out=None) for r, NR_bin in zip(bin_r_avg, NR)] # theoretically r values are supposed to be per point, but let's try it this way
    #         # scipy.special.jv(0, k * r, out=None)

    if plot:
        fig, ax = plt.subplots(2)
        ax[0].plot(bin_r_max, NR, label='NR')
        ax[0].plot(bin_r_max, NR_avg, label='NR_avg')
        ax[0].legend()
        
        ax[1].plot(bin_r_max, SNR, label='SNR')
        ax[1].plot(bin_r_max, S0R, label='S0R')
        ax[1].legend()
        
        plt.show()



    plt.imshow(NkR)
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

def structure_factor_from_g_r_fft(rdf_vals: np.ndarray,
                                  r_vals: np.ndarray,
                                  density,
                                  plot = True):
    dr = np.diff(r_vals)[0] # sampling rate
    df = 1 / dr # sample spacing
    k_values = np.fft.rfftfreq(r_vals.shape[0], d=dr)

    s_values = np.zeros(k_values.shape[0])
    for i, k in enumerate(k_values):
        ssum = 0
        for r, rdf in zip(r_vals, rdf_vals):
            ssum += jv(0, k*r) * (rdf-1) * r
        s_values[i] = 1 + 2 * np.pi * density * ssum * dr
    
    if plot:
        plt.plot(k_values, s_values)
        plt.xlabel("k")
        plt.ylabel("S(k)")
        plt.title("Structure factor")
        plt.show()

    return k_values, s_values


def psi_k(p, k):

    # alias for convenience
    vec_norm = np.linalg.norm

    N = p.shape[0] 

    knn = NearestNeighbors(n_neighbors=(k + 1)) # we look for one more neighbor bc it will find the same point...
    knn.fit(p)

    nei = knn.kneighbors(p, return_distance=False)[:,1:] # so we reject the point itself for each neighbor list

    vecs = np.empty((N, k, 2))
    for i in range(N):
        for n in range(k):
            vecs[i,n] = p[nei[i,n]] - p[i]

    psis = [] # = np.empty(N)
    angles = []

    for i in range(N):
         # vectors leading from p[i] to n[i][j]; shape (N, k, 2)
        # psis = np.empty(int((1+k)*k/2))
        # psis = []
        # we iterate over all neighbor pairs
        ii = 0
        for ni in range(k):
            for nj in range(ni+1,k):
                # psis[ii] = np.cos(
                angle = np.arccos(np.dot(vecs[i,ni], vecs[i,nj])/(vec_norm(vecs[i,ni]) * vec_norm(vecs[i,nj])))
                psis.append(np.cos(k*angle))
                angles.append(angle)
            #     ii += 1
            # ii += 1
        # psi[i] = np.mean(psis)
    
    return psis, angles
