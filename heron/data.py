"""

The data module is designed to load and prepare arbitrary data sets
for use in machine learning algorithms.

"""

import numpy as np
import copy

class Data():
    """
    The data class is designed to hold non-timeseries data, and is
    capable of automatically selecting test data from the provided 
    dataset.

    Future development will include the ability to add pre-selected 
    test and verification data to the object.
    """

    def __init__(self, targets, labels, target_sigma = None, label_sigma = None, target_names = None, label_names = None, test_targets=None, test_labels=None, test_size = 0.05):
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

        target_sigma : array-like
           Either an array of the uncertainty for each target point, or an 
           array of the uncertainties, as a float, for each column in the targets.

        label_sigma : array-like
           Either an array of the uncertainty for each target point, or an 
           array of the uncertainties, as a float, for each column in the labels.

        test_targets : array-like
           A set of test target data, which can be used to test the effectiveness of
           a prediction. If this isn't specified then the test data will be generated
           from the training data using the test_size parameter.

        test_labels : array-like
           The set of labels to accompany the test_targets.

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

        self.normaliser = {}
        #targets = np.atleast_2d(targets)
        #labels = np.atleast_2d(labels)
        self.targets = self.normalise(targets, "target")
        self.labels = self.normalise(labels, "label")
        
        # Prepare the sigmas
        # if target_sigma:
        #     # A full array of sigmas for each point
        #     if hasattr(target_sigma, '__len__') and (not isinstance(target_sigma, str)):
        #         if len(target_sigma) == len(targets):
        #             self.target_sigma = self.normalise(target_sigma, "target")
        #         else:
        #             raise ValueError("The length of the uncertainty array doesn't match the data")
        #     # An array with a fixed sigma for each column
        #     else:
        #         self.target_sigma = np.ones(len(target_sigma))*self.normalise(target_sigma, "target")
        # # If no sigma is provided, assume it equals zero
        # else:
        #     self.target_sigma = np.zeros_like(targets)
        # Do the same for the labels
        if label_sigma:
            # A full array of sigmas for each point
            if hasattr(label_sigma, '__len__') and (not isinstance(label_sigma, str)):
                if len(label_sigma) == labels.shape[0]:
                    self.label_sigma = self.normalise(label_sigma, "label")
                else:
                    
                    raise ValueError("The length of the label uncertainty array doesn't match the data")
            # An array with a fixed sigma for each column
            else:
                self.label_sigma = np.ones(len(labels))*self.normalise(label_sigma, "label")
        # If no sigma is provided, assume it equals zero
        else:
            self.label_sigma = np.zeros_like(labels)


        if not isinstance(test_targets, type(None)) and not isinstance(test_labels, type(None)):
            # Targets and labels have been provided, so we'll use those rather than
            # using the input data.
            self.test_targets = np.squeeze(self.normalise(test_targets, "target"))
            self.test_labels = np.squeeze(self.normalise(test_labels, "label"))
            self.labels = np.squeeze(self.labels)
        else:
            # Otherwise we use a portion of the training data.
            # Prepare the test entries
            test_entries = int(np.floor(test_size * len(self.labels)))
            test_entries = np.random.random_integers(0, len(self.labels)-1, test_entries)
            #
            self.test_targets = self.targets[test_entries]
            self.test_labels = self.labels[test_entries]
            #
            self.targets = np.delete(self.targets, test_entries, axis=0)
            self.labels = np.delete(self.labels, test_entries, axis=0)
            #self.target_sigma = np.delete(self.target_sigma, test_entries, axis=0)
            self.label_sigma = np.delete(self.label_sigma, test_entries, axis=0)

        if target_names:
            self.target_names = target_names
        else:
            self.target_names = range(self.targets.shape[-1])


        if label_names:
            self.label_names = label_names
        else:
            self.label_names = range(self.labels.shape[-1])

    def copy(self):
        """
        Return a copy of this data object.
        """
        return copy.copy(self)
            
    def name2ix(self, name):
        """
        Convert the name of a column to a column index.
        """
        n2i = {n:i for i, n in enumerate(self.target_names)}
        return n2i[name]

    def ix2name(self, name):
        """
        Convert the index of a column to a column name.
        """
        i2n = {i:n for i, n in enumerate(self.target_names)}
        return i2n[name]

    def calculate_normalisation(self, data, name):
        """
        Calculate the offsets for the normalisation. 
        We'll normally want to normalise the training data, and then be able 
        to normalise and denormalise new inputs according to that.

        Parameters
        ----------
        data : array-like
           The array of data to use to calculate the normalisations.
        name : str
           The name to label the constants with.
        """
        data = data
        dc = np.array(data.min(axis=0))
        range = np.array(np.abs(data.max(axis=0) - data.min(axis=0)))
        dc[range==0.0] = np.array(data.min(axis=0))[range==0]
        range[range==0.0] = 1.0

        self.normaliser[name] = (dc, range)
        return (dc, range)

    def get_starting(self):
        """
        Attempts to guess sensible starting values for the hyperparameter values.

        Returns
        -------
        hyperparameters : ndarray
           An array of values for the various hyperparameters.
        """
        #values = []
        #for ax in xrange(self.targets.shape[1]):
        #    values.append(np.median(np.unique(np.diff(self.targets[:, ax])))/2)
        #return np.array(values)
        return np.median(np.unique(np.diff(self.targets, axis=0), axis=0), axis=0).T[0]/2
    
    def normalise(self, data, name):
        """
        Normalise a given array of data so that the values of the data
        have a minimum at 0 and a maximum at 1. This improves the 
        computability of the majority of data sets.

        Parameters
        ----------
        data : array-like
           The array of data to be normalised.

        name : str
           The name of the normalisation to be applied, e.g. training or label

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
        data = np.array(data)
        if name in self.normaliser:
            dc, range = self.normaliser[name]
        else:
            dc, range = self.calculate_normalisation(data, name)

        if np.any(range) == 0.0:
            return data - dc
        else:
            normalised = (data - dc)
            normalised /= range

        return normalised
        

    def denormalise(self, data, name):
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
        if not name in self.normaliser:
            raise ValueError("There is no normalisation for {}".format(name))
        dc, range = self.normaliser[name]
        return data*range + dc
        #return data

    def add_data(self, targets, labels, target_sigma=None, label_sigma=None):
        """
        Add new rows into the data object.

        targets : array-like
           An array of training targets or "x" values which are
           to be used to train a machine learning algorithm.

        labels : array-like
           An array of training labels or "y" values which represent
           the observations made at the target locations of the data set.

        target_sigma : array-like
           Either an array of the uncertainty for each target point, or an 
           array of the uncertainties, as a float, for each column in the targets.

        label_sigma : array-like
           Either an array of the uncertainty for each target point, or an 
           array of the uncertainties, as a float, for each column in the labels.
        """
        targets = np.atleast_2d(targets)
        labels = np.atleast_2d(labels)
        if self.targets.shape[0]==1:
            self.targets = np.vstack([self.targets.T, self.normalise(targets, "target")]).T
        else:
            self.targets = np.vstack([self.targets, self.normalise(targets, "target")])

        if self.labels.shape[0] == 1:
            self.labels = np.vstack([self.labels.T, self.normalise(labels, "label")]).T
        else:
            self.labels = np.vstack([self.labels, self.normalise(labels, "label")])

        # Prepare the sigmas
        # if target_sigma:
        #     # A full array of sigmas for each point
        #     if hasattr(target_sigma, '__len__') and (not isinstance(target_sigma, str)):
        #         if len(target_sigma) == len(targets):
        #             if self.target_sigma.shape[0]==1:
        #                 self.target_sigma = np.vstack([self.target_sigma.T, self.normalise(target_sigma, "target")]).T
        #             else:
        #                 self.target_sigma = np.vstack([self.target_sigma, self.normalise(target_sigma, "target")])
        #         else:
        #             raise ValueError("The length of the uncertainty array doesn't match the data")
        #     # An array with a fixed sigma for each column
        #     else:
        #         if self.target_sigma.shape[0]==1:
        #             self.target_sigma = np.vstack([self.target_sigma.T, np.ones(len(target_sigma))*self.normalise(target_sigma, "target")]).T
        #         else:
        #             self.target_sigma = np.vstack([self.target_sigma, np.ones(len(target_sigma))*self.normalise(target_sigma, "target")])
        # # If no sigma is provided, assume it equals zero
        # else:
        #     if self.target_sigma.shape[0]==1:
        #         self.target_sigma = np.vstack([self.target_sigma.T, np.zeros_like(targets)]).T
        #     else:
        #         self.target_sigma = np.vstack([self.target_sigma, np.zeros_like(targets)])
        # Do the same for the labels
        if label_sigma:
            # A full array of sigmas for each point
            if hasattr(label_sigma, '__len__') and (not isinstance(label_sigma, str)):
                if len(label_sigma) == len(labels):
                    if self.label_sigma.shape[0]==1:
                        self.label_sigma = np.vstack([self.label_sigma.T, self.normalise(label_sigma, "label")]).T
                    else:
                        self.label_sigma = np.vstack([self.label_sigma, self.normalise(label_sigma, "label")])
                else:
                    raise ValueError("The length of the uncertainty array doesn't match the data")
            # An array with a fixed sigma for each column
            else:
                if self.label_sigma.shape[0]==1:
                    self.label_sigma = np.vstack([self.label_sigma.T, np.ones(len(label_sigma))*self.normalise(label_sigma, "label")]).T
                else:
                    self.label_sigma = np.vstack([self.label_sigma, np.ones(len(label_sigma))*self.normalise(label_sigma, "label")])
        # If no sigma is provided, assume it equals zero
        else:
            if self.label_sigma.shape[0]==1:
                self.label_sigma = np.vstack([self.label_sigma.T, np.zeros_like(labels)]).T
            else:
                self.label_sigma = np.vstack([self.label_sigma, np.zeros_like(labels)])



                
class Timeseries():
    """
    This is a class designed to hold timeseries data for machine
    learning algorithms.

    Timeseries data needs to be handled differently from other datasets
    as it is rarely likely to be advantageous to select individual points
    from a timeseries as either test data or verification data.
    Instead the timeseries class will select individual timeseries as the
    test and verification data.
    """

    def __init__(self, targets, labels, target_names = None, label_names = None, test_size = 0.05):
        """
        Construct the training data object with pre-loaded
        data.

        Parameters
        ----------
        targets : array-like
           An array of arrays of time-stamps for each timeseries.

        metadata : array-like
           An array of metadata for each timeseries.

        labels : array-like
           An array of data 

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

        In the case of timeseries data we expect the data to arrive with
        a fairly specific format:
        1) Time-stamps: each point in the timeseries must have an associated
                        time.
        2) Time-varying data: a series of data which are recorded at the 
                        specific time-stamped times.
        3) Metadata: information describing the observational or experimental
                        configuration which produced the data.
        For example, if we wished to learn about how temperatures varied over
        the year around some city, we could take measurements with a thermometer 
        at various points in the city at various times. The temperatures would
        constittute the data, the timestamps would be the times each reading was
        made, and the metadata might include details like the coordinates where 
        the measurement was made.
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
