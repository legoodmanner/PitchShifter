import numpy as np
import soundfile
import pitchshift
from pdb import set_trace as bp
from get_params import get_params

# Default parameters
D_BLOCK_SIZE = 2048  # 4096
D_N_BLOCKS = 4
D_LENGTH_MULT = 1
D_PITCH_MULT = 1
D_F_FILTER_SIZE = 8 #8

T0 = 1
T1 = 3

# INPUT_FILENAME = 'alphabet.wav'
INPUT_FILENAME = 'en001a.wav'
OUTPUT_FILENAME = 'output.wav'

class FileProcessor:
    def __init__(
        self,
        filename,
        block_size=D_BLOCK_SIZE,
        n_blocks=D_N_BLOCKS,
        length_mult=D_LENGTH_MULT,
        pitch_mult=D_PITCH_MULT,
        f_filter_size=D_F_FILTER_SIZE,
    ):
        self.block_size = block_size
        self.n_blocks = n_blocks

        self.length_mult = length_mult
        self.pitch_mult = pitch_mult
        # Always enable formant correction if a formant pitch scale is given
        self.f_filter_size = f_filter_size

        self.in_shift = self.block_size // self.n_blocks
        self.out_shift = int(self.in_shift * self.length_mult)

        self.in_file = soundfile.SoundFile(filename)
        self.rate = self.in_file.samplerate # Sampling frequency

        self.total_blocks = np.ceil(self.in_file.frames / self.in_shift)
        self.out_length = int(self.total_blocks * self.out_shift + self.block_size)

        self.out_data = np.zeros((self.out_length, self.in_file.channels))

        self.pvc = [
            pitchshift.PitchShifter(
                self.rate,
                self.block_size,
                self.pitch_mult,
            )
            for _ in range(self.in_file.channels)
        ]

    def run(self):
        t = 0
        params = get_params()

        for block in self.in_file.blocks(
            blocksize=self.block_size,
            overlap=(self.block_size - self.in_shift),
            always_2d=True,
        ):
            if block.shape[0] != self.block_size:
                block = np.pad(block, ((0, self.block_size - block.shape[0]), (0, 0)))
            for channel in range(self.in_file.channels):
                for param in params:
                    if t/self.rate > param[0] and t/self.rate < param[1]:
                        self.pitch_mult = param[2]
                        self.pvc[channel].pitch_mult = param[2]
                        break
                    else:
                        self.pitch_mult = D_PITCH_MULT
                        self.pvc[channel].pitch_mult = D_PITCH_MULT
                out_block = self.process_block(block[:, channel], channel)
                self.out_data[t : t + out_block.size, channel] += out_block

                
                # for param in params:
                #     if t/self.rate > param[0] and t/self.rate < param[1]:
                #         self.pitch_mult = param[2]
                #         self.pvc[channel].pitch_mult = param[2]
                #         out_block = self.process_block(block[:, channel], channel)
                #         self.out_data[t : t + out_block.size, channel] += out_block
                    # else :
                    #     self.pitch_mult = 1.5
                    #     self.pvc[channel].pitch_mult = 1.5
                    #     out_block = self.process_block(block[:, channel], channel)
                    #     self.out_data[t : t + out_block.size] = block
                # for param in params:
                #     if t/self.rate < param[0] or t/self.rate > T1:
                #         # self.out_data[t : t + out_block.size, channel] = block[:, channel]
                #         self.out_data[t : t + out_block.size] = block
            t += self.out_shift

        self.in_file.close()

        self.out_data = self.out_data / np.max(np.abs(self.out_data))

    def process_block(self, block, channel):
        out_block = self.pvc[channel].process(block, self.in_shift, self.out_shift)
        return out_block

    def write(self, filename):
        soundfile.write(filename, self.out_data, self.rate)


if __name__ == "__main__":

    processor = FileProcessor(INPUT_FILENAME)
    processor.run()
    processor.write(OUTPUT_FILENAME)


# 2switch
# t0 : 5, t1 : 10, p_mult : 0.5
# t0 : 15, t1 : 20, p_mult : 1.5

# slowlyIncreasing
# t0 : 5, t1 : 8, p_mult : 0.5
# t0 : 8, t1 : 11, p_mult : 0.6
# t0 : 11, t1 : 14, p_mult : 0.7
# t0 : 14, t1 : 17, p_mult : 0.8
# t0 : 17, t1 : 20, p_mult : 0.9
# t0 : 20, t1 : 23, p_mult : 1
# t0 : 23, t1 : 26, p_mult : 1.1
# t0 : 26, t1 : 29, p_mult : 1.2
# t0 : 29, t1 : 32, p_mult : 1.3
# t0 : 32, t1 : 35, p_mult : 1.4
# t0 : 35, t1 : 38, p_mult : 1.5