import numpy as np
import pandas as pd

def hex_crystal_network(size=30,
                    horizontal_spacing=1):
    n_rows = size
    assert(n_rows % 2 == 0) # validator
    n_columns = size
    a = horizontal_spacing # horizontal spacing
    h = np.sqrt(3) * a / 2 # vertical spacing

    points = []
    for i in range(n_rows): 
        y = i * h  
        for j in range(n_columns):
            x = j * a + (i % 2) * (a / 2)
            points.append([x,y])

    return pd.DataFrame(points, columns=["x","y"])
