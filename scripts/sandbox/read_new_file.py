from heron.data import DataWrapper


data = DataWrapper("test_file.h5")

print(data.get_training_data(label="mixed_model")[0])


