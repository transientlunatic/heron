"""
This file contains objects which provide the specifically-GW
parts of the surrogate models.
"""
from elk.waveform import Waveform, Timeseries
import astropy.constants as c
from math import log

class HofTSurrogate(object):
    def mean(self, p, times):
        """
        Return the mean waveform at a given location in the 
        BBH parameter space.
        """

        points = self._generate_eval_matrix(p, times)
        
        mean, var = self.gp.predict(self.training_data[:,self.c_ind['h+']],
                                    points,
                                    return_var=True,
        )
        mean_x, var = self.gp.predict(self.training_data[:,self.c_ind['hx']],
                                    points,
                                    return_var=True,
        )
        return Timeseries(data=mean/1e19, times=points[:,self.c_ind['time']]), \
            Timeseries(data=mean_x/1e19, times=points[:,self.c_ind['time']])


    def bilby(self, times, mass_1, mass_2, luminosity_distance):
        """
        Return a waveform from the GPR in a format expected by the Bilby ecosystem
        """
        
        times *= 100
    
        total_mass_cat = self.total_mass
        time_factor_cat = (c.c.value**3 / c.G.value)/(total_mass_cat*c.M_sun.value) #*1e4
        #h_factor = c.pc.value
        if mass_1 > mass_2:
            mass_ratio = log(mass_2/mass_1)
        else:
            mass_ratio = log(mass_1/mass_2)
        total_mass = (mass_1+mass_2)#*c.M_sun.value
    
        time_factor = (c.c.value**3 / c.G.value)/(total_mass*c.M_sun.value) 
    
        times *= (total_mass_cat / total_mass) #(time_factor/time_factor_cat)
    
        mean = self.mean(p={'mass ratio': mass_ratio}, times = times)
    
        return {"plus": mean[0].data / luminosity_distance , "cross": mean[1].data / luminosity_distance}


class BBHSurrogate(object):
    problem_dims = 8
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

class BBHNonSpinSurrogate(object):
    problem_dims = 2
    columns = {0: "time",
               1: "mass ratio",
               8: "h+",
               9: "hx"
    }
    c_ind = {j:i for i,j in columns.items()}
