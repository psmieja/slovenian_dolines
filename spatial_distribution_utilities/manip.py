import pandas as pd
import numpy as np

def perturb_points(points, alpha=0.1):
    # not the best way, but whatever...
    points.x = points.x.apply(lambda a: a + np.random.normal(scale=alpha))
    points.y = points.y.apply(lambda a: a + np.random.normal(scale=alpha))

    return points

