import numpy as np
import librosa
import math
import re
import os
from multiprocessing import Pool
from keras.utils import to_categorical

gender_dict = {'female':0, 'male':1}
region_dict = {'north':0, 'central':1, 'south':2}

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def getfeature(fname):
    timeseries_length=128
    hop_length = 512
    data = np.zeros((timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(fname)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    filelength = timeseries_length if mfcc.shape[1] >= timeseries_length else mfcc.shape[1]
    

    data[-filelength:, 0:13] = mfcc.T[0:timeseries_length, :]
    data[-filelength:, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[-filelength:, 14:26] = chroma.T[0:timeseries_length, :]
    data[-filelength:, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    return data

def processtrain(fname):
    data = getfeature(fname)
    gender, region = fname.split('/')[-2].split('_')
    print(fname)

    return data, gender, region

def processtest(fname):
    data = getfeature(fname)
    name = fname.split('/')[-1]
    print(fname)

    return data, name

def train():
    files = list(absoluteFilePaths('../data/voice_zaloai/train/'))
    p = Pool(40)    
    data = p.map(processtrain, files)
    X = [data[i][0] for i in range(len(data))]
    X = np.asarray(X)

    gender = [gender_dict[data[i][1]] for i in range(len(data))]
    gender = to_categorical(gender)

    region = [region_dict[data[i][2]] for i in range(len(data))]
    region = to_categorical(region)
    
    np.savez('../data/voice_zaloai/train', X=X, gender=gender, region=region)

def test():
    files = list(absoluteFilePaths('../data/voice_zaloai/public_test/'))
    p = Pool(40) 
    data = p.map(processtest, files)
    
    X = [data[i][0] for i in range(len(data))]
    X = np.asarray(X)
    
    name = [data[i][1] for i in range(len(data))]
    np.savez('../data/voice_zaloai/publictest', X=X, name=name)

if __name__=='__main__':
    train()
    test() 
