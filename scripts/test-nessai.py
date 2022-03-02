from heron.models.torchbased import HeronCUDA, train
import numpy as np
import matplotlib.pyplot as plt
import torch

generator = model = HeronCUDA(datafile="test_file_2.h5", datalabel="IMR training", device="cuda")
model.eval()


import gpytorch
with gpytorch.settings.max_cg_iterations(1500):
    train(model, iterations=10)


noise = 5e-19*torch.randn(1000)
signal = generator.time_domain_waveform(times=np.linspace(-.05, 0.005, 200), p={"mass ratio":0.7})

from elk.waveform import Timeseries

detection = Timeseries(data=(torch.tensor(signal[0].data) + noise), times=signal[0].times)

plt.plot(detection.times, detection.data)
plt.plot(signal[0].times, signal[0].data)


l = CudaLikelihood(generator, 
                   data = detection,
                   psd=noise.cuda().rfft(1),
                  )
#%%timeit
masses = np.linspace(0.5,1.0,200)
likes = [l({'mass ratio': m}) for m in masses]

print(masses[np.argmax(likes)])

