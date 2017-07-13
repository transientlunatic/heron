from scipy import stats

class Prior(object):
    """
    A prior probability distribution.
    """
    pass

class Normal(Prior):
    """
    A normal prior probability distribution.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        distro = stats.norm(self.mean, self.std)
        
    def logp(self, x):
        return distro.logpdf(x)
