"""
This file contains objects which provide the specifically-GW
parts of the surrogate models.
"""

class BBHSurrogate(object):
    columns = {0: "time",
                    1: "mass ratio",
                    2: "spin 1x",
                    3: "spin 1y",
                    4: "spin 1z",
                    5: "spin 2x",
                    6: "spin 2y",
                    7: "spin 2z",
                    8: "h+",
                    9: "hx"
            }
    c_ind = {j:i for i,j in columns.items()}
