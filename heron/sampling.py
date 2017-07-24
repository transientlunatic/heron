"""
Code to simplify sampling the Gaussian process.

"""

import numpy as np

class UndefinedParameters(Exception):
    pass

class UnsupportedError(Exception):
    pass

def draw_samples(gp, **kwargs):
    """
    Construct an array to pass to the Gaussian process to pull out a number of samples from a 
    high dimensional GP. 

    Parameters
    ----------
    gp : GPR object
       The Gaussian process object
    
    kwargs : int, list, or tuple
       The ranges for each value.
       In the format
         parameter = 0.9
       the axis will be constant 
       Alternatively a list or tuple can be passed to form a range (which uses the same
       arrangement of arguments as numpy's linspace function:
         parameter = (0.0, 1.5, 100)
       will produce an axis 100 samples long.

    """
    parameters = set(kwargs.keys())
    gp_parameters = set(gp.training_object.target_names)
    
    if len(parameters.difference(gp_parameters))>0:
        raise UndefinedParameters
    
    column_values = {}
    dimensions = 0
    len_vec = 1
    rangecols = []
    for parameter in kwargs.items():
        name = parameter[0]
        value = parameter[1]
        column = gp.training_object.name2ix(name)

        
        # make all of the axes for the sampling
        if isinstance(value, float) or isinstance(value, int):
            # The passed value is a single number, so it should be repeated for every
            # sample
            column_values[column] = value

        #elif "__iter__" in dir(value):
        else:
            # The passed value is something like a list, so we'll use it to construct an
            # axis using np.linspace
            if len_vec > 1 and dimensions >= 2:
                raise UnsupportedError
            if len(value) == 2:
                column_values[column] = np.linspace(value[0], value[1])
            elif len(value) == 3:
                column_values[column] = np.linspace(value[0], value[1], value[2])
            rangecols.append(column)
            len_vec = len(column_values[column])
            dimensions += 1
            
    if dimensions <= 1:
            
        output = np.zeros( (len_vec, len(kwargs.items())))

    elif dimensions == 2:
        gridpoints = np.meshgrid(column_values[rangecols[0]], column_values[rangecols[1]])
        points = np.dstack(gridpoints).reshape(-1, 2)
        
        column_values[rangecols[0]] = points[:,0]
        column_values[rangecols[1]] = points[:,1]
        
        output = np.zeros( (len(points), len(kwargs.items())))

    for column in column_values.items():
        output[:, column[0]] = column[1]
        
        
    return output
        
