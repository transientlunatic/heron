"""

The data module is designed to load and prepare arbitrary data sets
for use in machine learning algorithms.

"""

import numpy as np

class Data():
    def __init__(self, targets, labels, target_names = None, label_names = None, test_size = 0.05):
        """
        Construct the training data object with pre-loaded
        data.

        Parameters
        ----------
        targets : array-like
           An array of training targets or "x" values which are
           to be used to train a machine learning algorithm.

        labels : array-like
           An array of training labels or "y" values which represent
           the observations made at the target locations of the data set.

        test_size : float
           The size of the test set as a percentage of the whole data set.
           The test set is selected at random from the data, and is not
           provided to the algorithm as training data.

        Notes
        -----
        Data used in machine learning algorithms is usally
        prepared in sets of "targets", which are the locations
        at which observations are made in some parameter
        space, and "labels", which are the observations made at
        those target points.

        The variable names used in this package will attempt to
        use this convention where practical, although it is not uncommon
        to see targets and labels described as "x" and "y" repsectively
        in the literature, consistent with more traditional methods
        of data analysis.
        """
        targets = np.atleast_2d(targets)
        self.targets, self.targets_scale = self.normalise(targets)
        self.labels, self.labels_scale = self.normalise(labels)

        test_entries = np.floor(test_size * len(self.labels))
        test_entries = np.random.random_integers(0, len(self.labels), test_entries)
        #
        self.test_targets = self.targets[test_entries]
        self.test_labels = self.labels[test_entries]
        #
        self.targets = np.delete(self.targets, test_entries, axis=0)
        self.labels = np.delete(self.labels, test_entries, axis=0)

        if target_names:
            self.target_names = target_names
        else:
            self.target_names = range(self.targets.shape[-1])


        if label_names:
            self.label_names = label_names
        else:
            self.label_names = range(self.labels.shape[-1])

    def normalise(self, data):
        """
        Normalise a given array of data so that the values of the data
        have a minimum at 0 and a maximum at 1. This improves the 
        computability of the majority of data sets.

        Parameters
        ----------
        data : array-like
           The array of data to be normalised.

        Returns
        -------
        norm_data : array-like
           An array of normalised data.
        scale_factors : array-like
           An array of scale factors. The first is the DC offset, while
           the second is the multiplicative factor.

        Notes
        -----
        In order to perform the normalisation we need two steps:
        1) Subtract the "DC Offset", which is the minimum of the data
        2) Divide by the range of the data
        """
        if data.shape[0]==1:
            dc = data.min()
            range = data.max() - data.min()
        else:
            dc = data.min(axis=0)
            range = data.max(axis=0) - data.min(axis=0)

        normalised = (data - dc) / range

        return normalised, (dc, range)
        

    def denormalise(self, data, scale):
        """
        Reverse the normalise() method's effect on the data, and return it
        to the correct scaling.

        Parameters
        ----------
        data : array-like
           The normalised data
        scale : array-like
           The scale-factors used to normalise the data.

        Returns
        -------
        array-like
           The denormalised data
        """
        dc, range = scale
        return data*range + dc
