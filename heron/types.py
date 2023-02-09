class PSD:
    def __init__(self, data, frequencies):
        self.data = data
        self.frequencies = frequencies
        self.df = frequencies[1] - frequencies[0]
