import numpy as np

import phasevocoder
from pdb import set_trace as bp

NEW_FREQ = 50

class PitchShifter(phasevocoder.PhaseVocoder):
    """
    Pitch-shifts the input signal by warping the frequency spectrum
    """

    def __init__(
        self,
        samplerate,
        blocksize,
        pitch_mult,
    ):
        super().__init__(samplerate, blocksize)
        self.indices = np.arange(self.fft_size)
        self.pitch_mult = pitch_mult

    def process(self, block, in_shift, out_shift):
        magnitude, _, frequency = self.analyze(block, in_shift)

        if self.pitch_mult != 1:
            magnitude = np.interp(self.indices / self.pitch_mult, self.indices, magnitude, 0, 0)
            frequency = (
                np.interp(self.indices / self.pitch_mult, self.indices, frequency, 0, 0) * self.pitch_mult
            )

        out_block = self.synthesize(magnitude, frequency, out_shift)
        return out_block
