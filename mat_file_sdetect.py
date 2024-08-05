# -*- coding: utf-8 -*-

### ----------------------- Imports ----------------------- ###
import os
import h5py
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from helper import features
from sklearn.preprocessing import StandardScaler
from user_gui.verify_gui import VerifyGui
### ------------------------------------------------------- ###

# basic functions
def read_mat(file_path):
    """
    Convert single channel matlab file from Spike software to EDF

    Parameters
    ----------
    load_path : Str, path to mat file folder

    Returns
    -------
    signal
    fs

    """

    # load matlab file
    mat_read = h5py.File(file_path, mode='r')
    
    # get data
    struct_name = list(mat_read.keys())
    struct_name.remove('file')
    struct_name = struct_name[0]
    
    # get structure name
    sig = mat_read[struct_name + '/values'][0]
    
    # get sampling rate
    fs = 1/mat_read[struct_name + '/interval'][0]
    fs = int(fs[0])
    return sig, fs

def clean_signal(data, threshold=25):
    """
    Removes outliers and replaces with median

    Parameters
    ----------
    data : 1D signal, numpy array
    threshold : Real number, The default is 25.

    Returns
    -------
    clean_data : 1D signal, numpy array

    """
    clean_data = np.copy(data)
    clean_data = clean_data - np.mean(clean_data)
    idx = np.where(np.abs(data) > (threshold * np.std(data)))
    clean_data[idx] = np.median(clean_data)
    
    return clean_data

def butter_highpass(data, cutoff, fs, order=5):
    """
    High pass filter data

    Parameters
    ----------
    data : 1d ndarray, signal 
    cutoff : Float, cutoff frequency
    fs : Int, sampling rate
    order : Int, filter order. The default is 5.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    nyq = 0.5 * fs               # Nyquist Frequency (Hz)
    normal_cutoff = cutoff / nyq # Low-bound Frequency (Normalised)
    z,p,k = sp.signal.butter(order, normal_cutoff, btype='high', analog=False, output ='zpk') 
    sos = sp.signal.zpk2sos(z,p,k)         # Convert to second order sections
    y = sp.signal.sosfiltfilt(sos, data)   # Filter data
    return y

def clean_predictions(vector):
    """
    Performs dilation followed by erosion on a 1D binary vector.
    
    This function first applies binary closing to merge `1`s that are separated
    by up to two `0`s. Then, it performs binary opening to remove isolated `1`s
    surrounded by `0`s.
    
    Parameters:
    vector (numpy.ndarray): Input 1D binary vector.
    
    Returns:
    numpy.ndarray: The processed vector after dilation and erosion.
    """
    padded_vector = np.concatenate(([0], vector, [0]))
    closed = ndimage.binary_closing(padded_vector, structure=[1, 1, 1]).astype(int)
    opened = ndimage.binary_opening(closed, structure=[1, 1]).astype(int)
    return opened[1:-1]


def get_szr_idx(pred_array):
    """
    Identify seizure events and return their start and stop indices.
    
    Parameters
    ----------
    pred_array : numpy.ndarray, 1D boolean array where `True` indicates a seizure event.
        
    Returns
    -------
    idx_bounds : numpy.ndarray, 2D array of shape (n_events, 2) containing start and stop indices of valid events.
    
    Examples
    --------
    >>> pred_array = np.array([False, False, False, True, True, False, False])
    >>> find_szr_idx(pred_array)
    array([[3, 4]])
    """
    
    ref_pred = np.concatenate(([0], pred_array, [0]))
    transitions = np.diff(ref_pred)
    rising_edges = np.where(transitions == 1)[0]
    falling_edges = np.where(transitions == -1)[0] - 1
    idx_bounds = np.column_stack((rising_edges, falling_edges))

    return idx_bounds


# higher level functions
def get_features_single_ch(data, param_list):
    """
    get one channel features metrics 

    Parameters
    ----------
    data : 2D ndarray (1d = segments, 2d = time)
    param_list : ndarray with functions that extract single channgel parameters

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : list, feature names
    """
    
    # create data array
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list.
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list   
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:]) # extract feature
            
    return x_data, feature_labels


def preprocess(sig, down_factor, new_fs, cols):
    # preprocess and reshape
    sig = sp.signal.decimate(sig, down_factor)
    sig = clean_signal(sig, down_factor)
    sig = butter_highpass(sig, 2, new_fs)
    sig = sig[:cols*(len(sig)//cols)]
    sig = sig.reshape((-1, cols,))
    return sig



if __name__ == '__main__':
    
    # settings
    load_path = input("Please enter path to .mat file:\n")
    load_path = load_path.strip('"')
    if os.path.isfile(load_path):
        save_path = load_path[:-3] + 'csv'
        new_fs = 100
        win = 5
        sel_features = np.array((features.line_length, features.psd))
        thresholds = (4, 2.5)
        
        # preprocess and reshape
        sig, fs = read_mat(load_path)
        down_factor = round(fs/new_fs)
        cols =  int(win * new_fs)
        clean_sig = preprocess(sig, down_factor, new_fs, cols)
        
        # get features and predictions
        x_data, feature_names = get_features_single_ch(clean_sig, sel_features)
        x_data = StandardScaler().fit_transform(x_data)
        
        # get predictions (popular vote)
        y_pred_array = (x_data > np.array(thresholds))
        y_pred = np.sum(y_pred_array, axis=1)/len(sel_features)
        y_pred = y_pred > 0.5
        
        # clean predictions and find seizure index
        clean_pred = clean_predictions(y_pred)
        idx_bounds = get_szr_idx(clean_pred)
    
        # check for zero seizures otherwise proceed with gui creation
        if idx_bounds.shape[0] == 0:
            np.savetxt(save_path, np.zeros(y_pred.shape[0]), delimiter=',',fmt='%i')
            print(f'Zero seizures detected. Verified predictions were saved to {save_path}\n')
        else:
            data = clean_sig.reshape((clean_sig.shape[0], clean_sig.shape[1], 1))
            VerifyGui(data, idx_bounds, win, fs, save_path)
    else:
        print(f"Error. Please enter a valid path got this instead '{load_path}'\n")

