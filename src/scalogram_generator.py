import numpy as np
import pywt
from skimage.transform import resize

def generate_scalogram(beat, fs=360, img_size=(128,128)):
    scales = np.arange(1, 128)
    coef,_ = pywt.cwt(beat, scales, 'morl', sampling_period=1/fs)
    scalo = np.abs(coef)
    scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)
    return resize(scalo, img_size, mode='constant')
