
class LikelihoodUnderlay(unittest.TestCase):
    def setUp(self):
        self.model = HeronCUDAIMR(device=self.device)
        self.window = torch.blackman_window(180, device=self.device)
        self.empty = Timeseries(data=torch.zeros(len(self.window), device=self.device),
                                times=np.linspace(0,1, len(self.window))
        )
        self.asd = torch.ones(size=(len(self.window),2), dtype=torch.cdouble, device=self.device)
        self.likelihood = l = CUDALikelihood(self.model, 
                                             data = self.empty,
                                             psd=self.asd,
                                             window=torch.ones,
                                             detector_prefix="H1",
        )

    def test_model_call(self):
        """Check that the model is called correctly."""
        p = {"mass ratio": 0.9}
        result = self.model.time_domain_waveform(p,
                                                 times=np.linspace(0,1,len(self.window)))
        result_fft = torch.tensor(self.window*result[0].data).rfft(1)

        npt.assert_array_almost_equal(self.likelihood._call_model(p)[0].tensor, result_fft[1:]*1e19)

class TestLikelihoodGPU(LikelihoodUnderlay):
    def setUp(self):
        self.device = torch.device("cuda")
        super().setUp()

        
class TestLikelihoodCPU(LikelihoodUnderlay):
    def setUp(self):
        self.device = torch.device("cpu")
        super().setUp()
