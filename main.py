import numpy as np
import soundfile
from modules.pitchshift import PitchShifter
from modules.pitch_detection import track_pitch_mod, eval_pitchtrack_v2
from pdb import set_trace as bp
from get_params import get_params
import librosa
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os.path import join as pj
from os.path import basename
# Default parameters
D_BLOCK_SIZE = 2048  # 4096
D_N_BLOCKS = 4
D_LENGTH_MULT = 1
D_PITCH_MULT = 1
D_F_FILTER_SIZE = 8 #8

# T0 = 1
# T1 = 3

INPUT_FILENAME = 'data/constantSound1.wav'
OUTPUT_FILENAME = 'constantSound1_mod.wav'

class FileProcessor:
    def __init__(
        self,
        args,
        block_size=D_BLOCK_SIZE,
        n_blocks=D_N_BLOCKS,
        length_mult=D_LENGTH_MULT,
        pitch_mult=D_PITCH_MULT,
        f_filter_size=D_F_FILTER_SIZE,
    ):  
        self.args = args
        self.block_size = block_size
        self.n_blocks = n_blocks

        self.length_mult = length_mult
        self.pitch_mult = pitch_mult
        # Always enable formant correction if a formant pitch scale is given
        self.f_filter_size = f_filter_size

        self.in_shift = self.block_size // self.n_blocks
        self.out_shift = int(self.in_shift * self.length_mult)

        self.in_file = soundfile.SoundFile(self.args.audio_path)
        self.rate = self.in_file.samplerate # Sampling frequency

        self.total_blocks = np.ceil(self.in_file.frames / self.in_shift)
        self.out_length = int(self.total_blocks * self.out_shift + self.block_size)

        self.out_data = np.zeros((self.out_length, self.in_file.channels))

        self.pvc = [
            PitchShifter(
                self.rate,
                self.block_size,
                self.pitch_mult,
            )
            for _ in range(self.in_file.channels)
        ]

    def run(self):
        t = 0
        params = get_params(self.args.p_mults)

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
            t += self.out_shift

        self.in_file.close()

        self.out_data = self.out_data / np.max(np.abs(self.out_data))

    def process_block(self, block, channel):
        out_block = self.pvc[channel].process(block, self.in_shift, self.out_shift)
        return out_block

    def write(self, output_path=None):
        soundfile.write(
            output_path or pj(self.args.output_dir,'wavs', basename(self.args.audio_path).replace('.wav', '_mod.wav')), 
            self.out_data, 
            self.rate,
        )
    
    def evaluate(self):
        blockSize = 2048
        hopSize = blockSize // 2
        y, sr = librosa.load(INPUT_FILENAME, sr=self.rate)
        assert sr == self.rate, (sr, self.rate)
        shifted_pitch, _ = track_pitch_mod(self.out_data, blockSize=blockSize, hopSize=hopSize, fs=self.rate, med_kernel_size=15, voicingThres=-15)
        origin_pitch, _  = track_pitch_mod(y, blockSize=blockSize, hopSize=hopSize, fs=self.rate, med_kernel_size=15)
        shifted_origin_pitch = np.array(origin_pitch).copy()
        params = get_params(self.args.p_mults) # iters of tuple (t1, t2, ratio)
        for (t1, t2, ratio) in params:
            f1, f2 = librosa.time_to_frames([t1, t2], sr=self.rate, hop_length=hopSize)
            shifted_origin_pitch[f1:f2] *= ratio
        rmse, pfp, pfn = eval_pitchtrack_v2(shifted_pitch, shifted_origin_pitch)
        # visialize
        plt.plot(shifted_origin_pitch, label='shifted origin pitch')
        plt.plot(shifted_pitch, label='shifted pitch')
        plt.plot(origin_pitch, label='origin pitch')
        plt.legend()
        plt.savefig(pj(args.output_dir,'plots', basename(self.args.audio_path).replace('.wav', '.png')))
        print( rmse, pfp, pfn)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--p_mults', required=True, type=str, help='The path of the txt file for p_mult' )
    parser.add_argument('--audio_path', required=True, type=str, help='The path of the input audio file' )
    parser.add_argument('--output_dir', required=True, type=str, help='The root of the directory the output audio files are', default='output' )
    args = parser.parse_args()
    processor = FileProcessor(args)
    processor.run()
    processor.write()
    processor.evaluate()


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