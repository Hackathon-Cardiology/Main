import os
import random
import sys
import numpy as np
from scipy.signal import resample
from scipy.signal import spectrogram

sys.path.append("./")

from extract_transform_load import extractor

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed:=42)

class ETL :
    def __init__(self,dir,name='Patient_1',num_samples=None,shuffle=False):
        self.dir = dir
        self.name = name
        self.paths = self.get_paths(dir=dir,name=name,shuffle=shuffle)
        self.X,self.y = self.load_from_paths(self.paths, num_samples=num_samples,
                     starting_point=0)

    def extract_transform_load(self):
        return self.X,self.y

    def get_paths(self,dir,name,seed=42,shuffle=False) :
        """

        :return: lists of interictal and preictal paths to put in a dataset
        """

        types = ['interictal_segment', 'preictal_segment']
        paths = []

        if not os.path.isdir('./input'):
            os.mkdir('./input')
            print(f'Add EEG data to the input folder and run this program again')
            quit()

        for root, dirs, files in os.walk(self.dir):
            for i, file in enumerate(files):
                if type(name) == str :
                    if not name in file : continue
                    if not file.endswith('.mat') : continue

                    path = os.path.join(root, file)
                    segment = path[:-9]

                    if segment.endswith(types[0]) or segment.endswith(types[1]):
                        paths.append(path)
                        continue

                else : # then name is a list
                    for n in name :
                        if n in file :
                            path = os.path.join(root, file)
                            segment = path[:-9]

                            if segment.endswith(types[0]) or segment.endswith(types[1]):
                                paths.append(path)

                            continue

                # there are test file types with no answer for a kaggle competition, so skip them

        assert len(paths) > 0, f'No files found with name {name}'

        # shuffle paths
        if shuffle :
            random.Random(seed).shuffle(paths)

        return paths

    def load_from_paths(self, paths, num_samples=None, starting_point=0, seed=seed):
        """

        :param interictal_paths: list of interictal data
        :param preictal_paths: list of preictal data
        :param num_samples:
        :param num_samples: where to start taking from the list
        :param starting_point:
        :param seed:
        :param seed: random state for shuffle
        :return: pandas dataframe of electrode values and whether they are preictal
        """

        X = []
        Y = []

        if num_samples is None :
            num_samples = len(paths)

        # for each file, preprocess the file and add to dataset
        if starting_point >= len(paths) : return
        if starting_point + num_samples > len(paths) : num_samples = len(paths) - starting_point


        for i in range(starting_point,num_samples + starting_point) :
            path = paths[i]
            x,y = extractor(path)

            d_array = x[12,:]

            secs = len(d_array) / 5000  # Number of seconds in signal X
            samps = int(secs * 500)  # sample 500 times a second
            dsample_array = resample(d_array, samps)

            lst = list(range(300000))  # 3000000  datapoints initially

            span = 2000
            for m in lst[::span]:  # 5000 initial
                # make spectrograms every 2 seconds
                p_secs = dsample_array[m:m + span]  # d_array[0][m:m+15000]
                p_f, p_t, p_Sxx = spectrogram(p_secs, fs=500, return_onesided=False)
                p_SS = np.log1p(p_Sxx)
                arr = p_SS[:] / np.max(p_SS)
                X.append(arr)
                Y.append(y)


        num_classes = 2
        img_rows, img_cols = 256, 8

        X = np.array(X)
        X = np.array(X).reshape(X.shape[0], img_rows, img_cols, 1).astype('float32')
        y = np.array(Y)

        return X,y