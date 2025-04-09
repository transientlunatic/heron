"""
Testing for introducing a new data format based on hdf5 which we can use for all models in heron.

"""

import h5py

import pandas as pd

data = pd.read_csv("mixedmodel.dat", sep=" ", names=["source", "polarisation", "times", "mass ratio", "strain"])

data.loc[data['source']==0, 'source']="IMRPhenomPv2"
data.loc[data['source']==1, 'source']="SEOBNRv4"


data.loc[data['polarisation']==0, 'polarisation']="+"
data.loc[data['polarisation']==1, 'polarisation']="x"

with h5py.File("test_file.h5", "a") as f:

    file_format = f.create_group("version")
    file_format.create_dataset("version", data="0.0.1")
    
    training_data = f.create_group("training data")

    training_data.create_dataset("mixed_model/meta/parameters", data=["times", "mass ratio"])
    training_data.create_dataset("mixed_model/meta/polarisations", data=["+", "-"])
    training_data.create_dataset("mixed_model/meta/sources", data=["IMRPhenomPv2", "SEOBNRv4"])
    training_data.create_dataset("mixed_model/meta/reference mass", data=20)

    training_data.create_dataset("mixed_model/locations", data=data[['times', 'mass ratio']].to_numpy())
    training_data.create_dataset("mixed_model/polarisation", data=data['polarisation'].to_numpy())
    training_data.create_dataset("mixed_model/source", data=data['source'].to_numpy())

