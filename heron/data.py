"""
Common data format handling for heron models
"""

#import typing

import h5py
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt


class DataWrapper001:
    """The data wrapper for heron models using the v0.0.1 of the data
    format specification.

    :todo:`Make this compatible with gravitic as its development continues.`

    """

    def __init__(self,
                 filename: str,
                 write: bool = False):
        """Initialise a datawrapper using the 0.0.1 API.

        Parameters
        ----------
        filename : str
            The name of the file which should be loaded as a heron
            data object.
        write : bool
            Flag to determine if the file should be opened with
            read-write permissions. By defauls is False, in whcih case
            the file is opened read-only.

        Examples
        --------
        FIXME: Add docs.

        """
        self.specification = ""
        self.training_data = {}

        if write:
            mode = "a"
        else:
            mode = "r"

        self.h5file = h5py.File(filename, mode)

    def __getitem__(self, item):
        return self.h5file['training data'][item]

    
    @classmethod
    def create(cls, filename) -> 'DataWrapper001':
        """
        Create a new data file.
        This method simply creates the file in which the data is stored.
        It doesn't attempt to add or generate the data.

        Parameters
        ----------
        filename : str
            The name of the file which should be loaded as a heron
            data object.

        Examples
        --------

        In its very simplest use we can just create a file and do
        nothing with it.

        >>> from heron.data import DataWrapper
        >>> data = DataWrapper.create("test_file.h5")
        """
        with h5py.File(filename, "a") as f:

            file_format = f.create_group("version")
            file_format.create_dataset("version", data="0.0.1")

            _ = f.create_group("training data")

        return cls(filename, write=True)

    def _add_meta(self,
                  meta: str,
                  data: str,
                  group: str):
        """Add a new data source to the data file.

        Insert a new source of metadata to the data object. For normal
        purposes you shouldn't need to directly access this method.

        Parameters
        ----------
        meta : str
            The name of the metadata variable
        data : str
            The metadata to be stored.
        group : str
            The name of the group to apply the metadata to.

        Examples
        --------
        FIXME: Add docs.


        """
        if not isinstance(data, (list, np.ndarray)):
            data = [data]
        training_data = self.h5file["training data"]
        if f"{group}/meta/{meta}" in training_data:
            if data in training_data[f"{group}/meta/{meta}"]:
                pass
            else:
                training_data[f"{group}/meta/{meta}"].resize((training_data[f"{group}/meta/{meta}"].shape[0] + len(data) ), axis=0)
                training_data[f"{group}/meta/{meta}"][-len(data):] = data
        else:
            training_data.create_dataset(f"{group}/meta/{meta}",
                                         (len(data),), data=data, chunks=True, maxshape=(None, ))


    def _assign_integer_keys(self,
                             meta: str,
                             value,
                             group: str):

        if isinstance(value, str):
            value = value.encode("ascii")

        training_data = self.h5file["training data"]
        if f"{group}" in training_data:
            if meta in training_data[f"{group}"]["meta"]:
                if value in training_data[f"{group}"]["meta"][f"{meta}"]:
                    ixes = dict(enumerate(training_data[f"{group}/meta/{meta}"]))
                    ixes = dict((v,k) for k,v in ixes.items())
                    return int(ixes[value])
        else:
            raise Exception

    def add_data(self,
                 group: str,
                 polarisation: str,
                 reference_mass: float,
                 source: str,
                 parameters: list,
                 locations: np.ndarray,
                 data: np.ndarray):
        """
        Add a new data series to the data structure.

        This interface is designed for addition of bulk waveform data,
        and not individual waveforms.

        Parameters
        ----------
        group : str
            The data group within the object to store the new data
            series.
        polarisation : str
            The polarisation of the new data.
        reference_mass : float
            The reference total mass used to generate the data.
        source : str
            A string describing the source of the data series.
        parameters : list
            The names of the parameters which are represented in the
            data.
        locations : np.ndarray
            The parameter locations for each individual datum.
        data : np.ndarray
            The data itself.

        Examples
        --------
        Add data to a data structure from a CSV file.
        >>> from heron.data import DataWrapper
        >>> data = DataWrapper.create("test_file.h5")
        >>> input_data = pd.read_csv("mixedmodel.dat", sep=" ",
        ...                     names=["source", "polarisation", "times", "mass ratio", "strain"])
        >>> data.add_data(group="mixed_model",
        ...               polarisation="+",
        ...               reference_mass=20,
        ...               source="IMRPhenomXPHM",
        ...               parameters=["times", "mass ratio"],
        ...               locations=input_data['times', 'mass ratio'].to_numpy(),
        ...               data=input_data['strain'])
        """
        training_data = self.h5file["training data"]

        self._add_meta("parameters", parameters, group)
        self._add_meta("polarisations", polarisation, group)
        self._add_meta("sources", source, group)
        self._add_meta("reference mass", reference_mass, group)

        polarisation = np.ones(len(data)) * self._assign_integer_keys("polarisations",
                                                                      polarisation,
                                                                      group)
        source = np.ones(len(data)) * self._assign_integer_keys("sources",
                                                                source,
                                                                group)

        for i, parameter in enumerate(parameters):
            training_data.create_dataset(f"{group}/locations/{parameter}",
                                         (len(locations[:,i]),),
                                         data=locations[:,i],
                                         chunks=True,
                                         maxshape=(None, ))
        training_data.create_dataset(f"{group}/polarisation",
                                     data=polarisation,
                                     chunks=True,
                                     maxshape=(None))
        training_data.create_dataset(f"{group}/source",
                                     data=source, chunks=True, maxshape=(None))
        training_data.create_dataset(f"{group}/data",
                                     data=data, chunks=True, maxshape=(None))

    def add_waveform(self,
                     group: str,
                     polarisation: str,
                     reference_mass: float,
                     source: str,
                     locations: dict,
                     times: np.ndarray,
                     data: np.ndarray):
        """
        Add a single waveform to the training data.
        """
        training_data = self.h5file["training data"]

        for parameter, location in locations.items():
            if f"{group}/locations/{parameter}" not in training_data:
                training_data.create_dataset(f"{group}/locations/{parameter}",
                                             (len(data),),
                                             data=np.ones(len(data))*location,
                                             chunks=True,
                                             maxshape=(None,))
            else:
                training_data[f"{group}/locations/{parameter}"].resize((training_data[f"{group}/locations/{parameter}"].shape[0] + len(data) ), axis=0)
                training_data[f"{group}/locations/{parameter}"][-len(data):] = np.ones(len(data))*location

        if f"{group}/locations/time" not in training_data:
            training_data.create_dataset(f"{group}/locations/time",
                                         (len(times), ),
                                         data=times,
                                         chunks=True,
                                         maxshape=(None, ))
        else:
            training_data[f"{group}/locations/time"].resize((training_data[f"{group}/locations/time"].shape[0] + len(data) ), axis=0)
            training_data[f"{group}/locations/time"][-len(data):] = times

        if f"{group}/polarisation" not in training_data:
            training_data.create_dataset(f"{group}/polarisation",
                                         (len(times), ),
                                         data=len(data)*[str(polarisation)],
                                         chunks=True,
                                         maxshape=(None, ))
        else:
            training_data[f"{group}/polarisation"].resize((training_data[f"{group}/polarisation"].shape[0] + len(data) ), axis=0)
            training_data[f"{group}/polarisation"][-len(data):] = len(data)*[str(polarisation)]

        if f"{group}/data" not in training_data:
            training_data.create_dataset(f"{group}/data",
                                         (len(data), ),
                                         data=data,
                                         chunks=True,
                                         maxshape=(None, ))
        else:
            training_data[f"{group}/data"].resize((training_data[f"{group}/data"].shape[0] + len(data) ), axis=0)
            training_data[f"{group}/data"][-len(data):] = data
            
        self._add_meta("parameters", list(locations.keys()), group)
        self._add_meta("polarisations", polarisation, group)
        self._add_meta("sources", source, group)
        self._add_meta("reference mass", reference_mass, group)
        
        
    def get_training_data(self,
                          label: str,
                          polarisation: str = "+",
                          form: str = None):
        training_data = self.h5file["training data"]
        locations = list(training_data[label]["locations"].keys())

        iloc = np.array(training_data[label]["polarisation"]) == self._assign_integer_keys("polarisations",
                                                                                           polarisation,
                                                                                           label)
        
        xdata = np.zeros((len(locations), len(iloc)))
        ydata = np.array(training_data[f"{label}"]["data"][iloc])

        for i, location in enumerate(locations):
            xdata[i, :] = training_data[f"{label}"]["locations"][f"{location}"]

        if form is None:
            return xdata, ydata

    def plot_surface(self,
                     label: str,
                     x: str,
                     y: str,
                     polarisation: str = "+",
                     decimation: int = 1
                     ):
        training_data = self.h5file["training data"]
        fig, axis = plt.subplots(1, 1, dpi=300)

        iloc = np.array(training_data[label]["polarisation"]) == self._assign_integer_keys("polarisations",
                                                                                           polarisation,
                                                                                           label)
        axis.scatter(x=training_data[f"{label}"]["locations"][f"{x}"][iloc][::decimation],
                     y=training_data[f"{label}"]["locations"][f"{y}"][iloc][::decimation],
                     c=training_data[f"{label}"]["data"][iloc][::decimation],
                     marker='.')
        return fig

DataWrapper = DataWrapper001
