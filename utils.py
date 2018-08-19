import os
import numpy as np
from scipy.misc import imsave

def get_concat(input,gt,est):
    concat = np.concatenate([input, gt, est], axis = 1)
    concat = np.clip(np.power(concat, 1/2.2), 0, 1)
    return concat
