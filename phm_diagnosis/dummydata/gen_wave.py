import numpy as np
import matplotlib.pyplot as plt
import pdb

'''
contribute a wave with five elements
:param a: amplitude
:param f: freq
:param p: phase
:param l: lenth(sec)
:param s: sampling
'''


def to_show(x, y):
    plt.plot(x, y)
    plt.show()


def sin_wave(a, f, p, l, s):
    assert 0 <= p <= np.pi * 2
    cnt_seconds = int(l // 1.0)
    semi_seconds = l % 1.0
    base_wave = np.arange(p, p + np.pi * 2, np.pi * 2 / s)
    base_wave = np.concatenate([base_wave.repeat(
        cnt_seconds).reshape(-1, cnt_seconds).T.reshape(-1), base_wave[:int(s * semi_seconds)]])
    x = np.arange(len(base_wave)) / f
    y = np.sin(base_wave) * a
    return x, y


def cos_wave(a, f, p, l, s):
    assert 0 <= p <= np.pi * 2
    cnt_seconds = int(l // 1.0)
    semi_seconds = l % 1.0
    base_wave = np.arange(p, p + np.pi * 2, np.pi * 2 / s)
    base_wave = np.concatenate([base_wave.repeat(
        cnt_seconds).reshape(-1, cnt_seconds).T.reshape(-1), base_wave[:int(s * semi_seconds)]])
    x = np.arange(len(base_wave)) / f
    y = np.cos(base_wave) * a
    return x, y


def rec_wave(a, f, p, l, s):
    x, y = sin_wave(a, f, p, l, s)
    return x, [a if _ >= 0 else -a for _ in y]


'''
prob data generate
'''

def _normal_data(loc, scale, size):
    return np.random.normal(loc, scale, size)

def _uniform_data(low, high, size):
    return np.random.uniform(low, high, size)

def noise(flag:str, loc_or_low:float, scale_or_high:float):
    def outer(func):
        def inner(*args,**keargs):
            if flag == 'normal':
                res = func(*args, **keargs)
                assert isinstance(res[1], np.ndarray)
                y = _normal_data(loc_or_low, scale_or_high, len(res[1]))
                return res[0], res[1]+y 
            elif flag == 'uniform':
                res = func(*args, **keargs)
                assert isinstance(res[1], np.ndarray)
                y = _uniform_data(loc_or_low, scale_or_high, len(res[1]))
                return res[0], res[1]+y 
            else:
                res = func(*args,**keargs)
                return res
        return inner
    return outer

if __name__ == '__main__':

    '''
    to_show(*sin_wave(3,60,0,3,30))
    to_show(*cos_wave(3,60,0,3,30))
    to_show(*rec_wave(3,60,0,3,30))
    '''
