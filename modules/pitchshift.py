import numpy as np

from pdb import set_trace as bp

NEW_FREQ = 50
# TODO: Move peak detection flags to command-line args
MATCH_PEAKS = True
PEAK_MAX_DIST = 16
PEAK_THRESH = 0.01


class PhaseVocoder():
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.fft_size = (blocksize // 2) + 1

        self.last_phase = np.zeros(self.fft_size)
        self.last_phase_out = np.zeros(self.fft_size)

        self.window = np.hanning(blocksize)
        self.freq = np.fft.rfftfreq(blocksize, 1 / samplerate)

    def analyze(self, block, advance):
        in_block = block * self.window  # np.fft.fftshift()
        fft = np.fft.rfft(in_block)

        magnitude = np.abs(fft)
        phase = np.angle(fft)

        dt = advance / self.samplerate

        min_f = self.est_freqs_div(phase, dt)

        self.last_phase = phase

        return magnitude, phase, min_f

    def est_freqs_div(self, phase, dt):
        # TODO: This runs into problems at first bin, investigate

        freq_base = (phase - self.last_phase) / (2 * np.pi * dt)
        n = np.maximum(np.round((self.freq - freq_base) * dt), 0)
        min_f = freq_base + (n / dt)
        
        return min_f

    def constrain_phase(self, phase):
        return ((phase + np.pi) % (np.pi * 2)) - np.pi

    def synthesize(self, magnitude, frequency, advance):
        dt = advance / self.samplerate

        out_phase = self.last_phase_out + 2 * np.pi * frequency * dt

        out_phase = self.constrain_phase(out_phase)

        self.last_phase_out = out_phase

        fft = magnitude * np.exp(1j * out_phase)

        out_block = np.fft.irfft(fft) * self.window

        return out_block



class PitchShifter(PhaseVocoder):
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