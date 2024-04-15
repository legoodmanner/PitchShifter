import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann
from scipy.signal import medfilt

def median_filter(arr, kernel_size):
    if type(arr) is not list:
        arr = arr.tolist()  # check data type "list"

    length = len(arr)  # get length of input array
    part = (kernel_size - 1) // 2  # calc number of elements at (left, right) side
    med_arr = []  # array after median filter

    for i in range(length):
        left = i - part
        right = i + part
        if left < 0:  # out of index at left side
            tmp = [0] * (0 - left) + arr[0:right + 1]  # add 0 element to left side
        elif right >= length:  # out of index at right side
            tmp = arr[left:length] + [0] * (right - length + 1)  # add 0 element to right side
        else:
            tmp = arr[left: right + 1]
        tmp.sort()  # sort up ascending
        med_arr.append(tmp[part])  # add element after filter
    return np.array(med_arr)

def smooth_2d_cruve(s, n=3):
    if len(s.shape) == 1:
        s = s[np.newaxis,:]
    _s = np.pad(s, pad_width=((0,0),(n//2, n//2)), mode='edge')
    for i in range(s.shape[-1]):
        s[:,i] = np.mean(_s[:,i:i+n], -1)
    return s

def downsample(X, factor):
    n = 3
    res = np.zeros((X.shape[0], X.shape[-1]//factor))
    X = np.pad(X, pad_width=((0,0),(n//2, n//2)), mode='edge')
    for i in range(res.shape[-1]):
        res[:,i] = np.mean(X[:,i*factor:i*factor+n], -1)
    return res

def find_FFTpeak(mag, smooth=True):
    # return peak idx
    if len(mag.shape) == 1:
        mag = mag[np.newaxis,:]
    if smooth:
        mag = smooth_2d_cruve(mag)
    # min_index = np.argmax(np.diff(mag, axis=-1) > 0, -1)
    # # 2d np.arange check if > min_index
    # cond = np.array([list(range(mag.shape[-1]))]*mag.shape[0]) > min_index[:,np.newaxis]
    # mag = np.where(cond, mag, 0)
    # idx = np.argmax(mag, -1)
    # return idx+min_index+1 
    return np.argmax(mag, -1)
    
def getTimeInSec(n_blocks, hopSize, fs):
    return np.linspace(0, (n_blocks-1)*hopSize/fs, n_blocks)

def block_audio(x, block_size, hop_size, fs, return_time=False):
    # stereo -> mono
    if len(x.shape) > 1:
        x = np.mean(x, axis=0) if x.shape[0] <= 2 else np.mean(x, axis=1)
    # padding
    if not len(x) % hop_size and len(x):
        x = np.pad(x, (0, hop_size - len(x) % hop_size), 'constant', constant_values=0)
    n_blocks = len(x) // hop_size
    x = np.pad(x, (0, block_size - hop_size), 'constant', constant_values=0)
    timeInSample = np.array(list(range(0, int(n_blocks)))) * hop_size
    res = np.zeros((n_blocks, block_size))
    for i in range(n_blocks):
        res[i] = x[timeInSample[i]:timeInSample[i]+block_size]
    if not return_time:
        return res
    else:
        return res, timeInSample / fs

def compute_spectrogram(xb, fs):
    n_fft = xb.shape[-1]
    mag = abs(rfft(xb * hann(n_fft)))
    assert mag.shape[-1] == n_fft//2+1, print(mag.shape[-1])
    return mag, np.arange(n_fft//2+1) * fs / n_fft

# Duplicated from hw1
def comp_acf(input_vector, is_normalized):
    rxx = np.correlate(input_vector, input_vector, 'full')
    if is_normalized:
        rxx /= np.sum(np.power(input_vector,2))+0.00001
    return rxx[len(rxx)//2:]

def get_redundant_freqIndex(fs, source='all'):
    if source=='all':
        lowest_midi = 27.5  # A0
        highest_midi = 3000
    return int(fs/highest_midi), int(fs/lowest_midi)

def get_f0_from_acf(r, fs):
    low, high = get_redundant_freqIndex(fs)
    dr = np.diff(r)
    # max(highest frquency, first index of r'(x) > 1, normalized auto-correlation > thre_hold)
    min_index = np.max([low, np.argmax(dr>0), np.argmax(r>0.3)])
    p = np.argmax(r[np.arange(min_index+1, r.size)]) + 1
    return fs / (p+min_index+1)

def get_f0_from_Hps(X, fs, order):
    if len(X.shape) == 1:
       X = X[np.newaxis, :]
    n_fft = (X.shape[-1]-1) * 2
    res = X
    for m in range(2, order+1):
        # Plus 1?
        res[:, :int(X.shape[-1]//m)] *= (1 + downsample(X, factor=m))
    # res = smooth_2d_cruve(res)
    idx = np.argmax(res, -1)
    return idx * fs / n_fft

#### ========================  ####

def extract_rms(xb):
    if len(xb.shape) == 1:
       xb = xb[np.newaxis, :]
    # print(xb.shape)
    rms = np.sqrt(np.mean(np.square(xb), -1))
    return 10 * np.log10(rms/1)

def create_voicing_mask(rmsDb, thresholdDb):
    return np.where(rmsDb>=thresholdDb, np.ones_like(rmsDb), np.zeros_like(rmsDb))

def apply_voicing_mask(f0, mask):
    return np.where(mask, f0, 0)        

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    xb = block_audio(x, blockSize, hopSize, fs)
    mag, fInHz = compute_spectrogram(xb, fs)
    idx = find_FFTpeak(mag)
    maxInHz = fInHz[idx]
    return maxInHz, getTimeInSec(xb.shape[0], hopSize, fs)

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb = block_audio(x, blockSize, hopSize, fs)
    mag, fInHz = compute_spectrogram(xb, fs)
    f0s = get_f0_from_Hps(mag, fs, order=4)
    return f0s, getTimeInSec(xb.shape[0], hopSize, fs)

def track_pitch_acf(x, block_size, hop_size, fs):
    blocks, timeInSec = block_audio(x, block_size, hop_size, fs, return_time=True)
    #### parallelable? ####
    f0s = []
    for b in blocks:
        r = comp_acf(b, is_normalized=True)
        f0s += [get_f0_from_acf(r, fs)]
    return f0s, timeInSec

# -- EVAL --
def eval_voiced_fp(estimation, annotation):
    length = min(len(annotation), len(estimation))
    annotation, estimation = annotation[:length], estimation[:length]
    return np.sum(estimation[annotation==0]>0, -1) / np.sum(annotation==0, -1)

def eval_voiced_fn(estimation, annotation):
    length = min(len(annotation), len(estimation))
    annotation, estimation = annotation[:length], estimation[:length]
    return np.sum(estimation[annotation>0]==0, -1) / np.sum(annotation>0, -1)

def eval_pitchtrack_v2(estimation, annotation):
    # [errCentRms, pfp, pfn] 
    summ = 0
    n = len(estimation)
    for pred, target in zip(estimation, annotation):
        if target == 0 or pred==0:
            n -= 1
            continue
        cent = 1200 * np.log2(np.divide(pred+0.0001, target+0.0001))
        summ += cent**2
    rmse = np.sqrt(summ/n)
    pfn = eval_voiced_fn(estimation, annotation)
    pfp = eval_voiced_fp(estimation, annotation)
    return rmse, pfp, pfn

def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == 'acf':
        xb, timeInSec = block_audio(x, blockSize, hopSize, fs, return_time=True)
        f0s = []
        for xb_ in xb:
            r = comp_acf(xb_, is_normalized=True)
            f0s += [get_f0_from_acf(r, fs)]
        f0s = np.array(f0s)
    elif method == 'max':
        f0s, timeInSec = track_pitch_fftmax(x, blockSize, hopSize, fs)
    elif method == 'hps':
        f0s, timeInSec = track_pitch_hps(x, blockSize, hopSize, fs)
    else:
        raise RuntimeError(f'The method of {method} is not valid ')
    xb = block_audio(x, blockSize, hopSize, fs)
    mask = create_voicing_mask(extract_rms(xb), thresholdDb=voicingThres)
    f0s = apply_voicing_mask(f0s, mask)
    return f0s, timeInSec

def extract_spectral_crest(xb, fs):
    mag, _ = compute_spectrogram(xb, fs)
    return np.max(mag, axis=-1) / np.sum(mag, axis=-1)

def track_pitch_mod(x, blockSize, hopSize, fs, voicingThres=-20, win_len=7, mwin_len=7, med_kernel_size=99):
    xb = block_audio(x, blockSize, hopSize, fs)
    mask = create_voicing_mask(extract_rms(xb), thresholdDb=voicingThres)
    changes = abs(np.diff(mask))
    changes = np.pad(changes, pad_width=(win_len//2, win_len//2), mode='edge')
    crest = extract_spectral_crest(xb, fs)

    # Compute results from all the modes
    resMax, timeInSec = track_pitch_fftmax(x, blockSize, hopSize, fs)
    resHPS, _ = track_pitch_hps(x, blockSize, hopSize, fs)
    resACF, _ = track_pitch_acf(x, blockSize, hopSize, fs)
    # Result container
    f0s = np.zeros(len(xb))
    hpscount = 0
    maxcount = 0

    for i in range(len(xb)):
        if crest[i] > .3:
            f0s[i] = resMax[i]
            maxcount += 1
        elif changes[i:i+win_len+1].any():
            hpscount += 1
            f0s[i] = resHPS[i]
        else:
            f0s[i] = resACF[i]

    # tmp = np.append(np.array([False]), np.true_divide(f0s[1:], f0s[:-1]) > 2).nonzero()[0]
    # for t in tmp:
    #     f0s[t] = (f0s[t-1] + f0s[t+1] if t+1 < len(f0s) else f0s[t-1] ) / 2
    tmp = np.pad(f0s, pad_width=(mwin_len//2, mwin_len//2), mode='edge')
    for i in range(len(f0s)):
        if f0s[i] / np.median(tmp[i:i+mwin_len]) > 2:
            f0s[i] = np.median(tmp[i:i+mwin_len])
    f0s = median_filter(f0s, kernel_size=med_kernel_size)
    f0s = apply_voicing_mask(f0s, mask)
    # print(hpscount, maxcount)
    return f0s, timeInSec


if __name__ == '__main__':
    import soundfile as sf
    wav, sr = sf.read('hw3/corea1.wav')
    maxInHz, timeInSec = track_pitch_fftmax(wav, 1024, 512, 44100)
    print(maxInHz.shape)
    print(timeInSec[:10])
    print('end')
    # time resolution would be fs / hop 
