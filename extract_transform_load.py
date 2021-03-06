"""
Extract, transform, and load the data
run this script in a folder with an 'input/' folder, which contains the .mat files that you want to transform
"""

from scipy.io import loadmat

def extractor(path) :
    """

    :param path: path to the .mat file that you wish to extract
    :return: numpy array of the wave data. Data is expressed in integer form
    """
    data = loadmat(path)
    filename = path.split('\\')[-1]
    if not filename.startswith('Patient') or not filename.startswith('Dog') :
        filename = path.split('/')[-1]

    assert 'interictal' in filename or 'preictal' in filename, f'File must be interictal or preictal. ' \
                                                               f'You probably passed a test file. Your file: {path}'

    is_preictal = 1 if 'preictal' in filename else 0

    for k,v in data.items() :
        if k.startswith('interictal') or k.startswith('preictal') :
            return v[0][0][0],is_preictal

    return None

if __name__ == '__main__' :
    import numpy as np

    data = extractor('./input/Patient_1/Patient_1/Patient_1_preictal_segment_0001.mat')
    print(np.array(data[0]).shape)

    data = extractor('./input/Patient_2/Patient_2/Patient_2_interictal_segment_0001.mat')
    print(data[0].shape)
    data = extractor('./input/Patient_2/Patient_2/Patient_2_interictal_segment_0001.mat')
    print(data[0].shape)
    data = extractor('./input/Patient_2/Patient_2/Patient_2_interictal_segment_0001.mat')
    print(data[0].shape)