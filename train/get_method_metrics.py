# -*- coding: utf-8 -*-

### ------------------------------ IMPORTS ------------------------------- ###
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from train.feature_settings import param_list, cross_ch_param_list
from train.feature_settings import metrics, feature_labels, ch_list
from helper.io_getfeatures import get_data, get_features_allch
from helper.array_helper import find_szr_idx, match_szrs_idx
### ---------------------------------------------------------------------- ###

class MethodMetrics:
    """
    Get model metrics from different feature combinations.
    """

    def __init__(self, main_path, verified_dir,
                 data_dir='filt_data', csv_dir='train', 
                 parameter_csv='parameter_catalogue.csv',
                 output_csv_name='parameter_metrics.csv'):
        """
        
        Parameters
        ----------
        main_path : Str, path to parent directory.
        verified_dir : str, name of verified folder with binary csv files.
        data_dir : str, name of folder with h5 voltage data.
        csv_dir : str, directory to save output save.
        parameter_csv : str, name of parameter catalogue csv file.
        output_csv_name : str, name of output csv file.
        
        --- Examples ---
        MethodTest('\data\training_data', 'verified_predictions').multi_folder()
        """
        
        # pass parameters to object
        self.main_path = main_path
        self.verified_dir = verified_dir
        self.data_dir = data_dir
        self.parameter_csv_path = os.path.join(csv_dir, parameter_csv)
        self.output_csv_path = os.path.join(csv_dir, output_csv_name)
        
        # define bounds to remove short seizure segments
        self.dur = 1
        self.remove_bounds = np.array([0,1])

        # get paremeters from module
        self.ch_list = ch_list
        self.metrics = metrics
        self.feature_labels = list(feature_labels)
        
        # get parameter catalogue and ensure the metrics
        self.df = pd.read_csv(self.parameter_csv_path)
        self.df[self.metrics] = 0
        
        # get parameter index
        self.thresh = np.where(np.array(self.df.columns.str.contains('Thresh')))[0]
        self.weights = np.where(np.array(self.df.columns.str.contains('Weight')))[0]
        self.enabled = np.where(np.array(self.df.columns.str.contains('Enabled')))[0]
        
        # match order of dataframe features to feature_labels output
        df_features = self.df.columns[self.thresh].to_list()
        df_features = [x.replace('Thresh_', '') for x in df_features]
        self.idx = np.zeros(len(df_features), dtype=int)
        for i,feature in enumerate(df_features):
            self.idx[i] = self.feature_labels.index(feature)
        
        # get total time in hours
        self.duration = self.get_total_duration()/3600
        
    def get_total_duration(self):
        """
        Get total file duration in seconds.

        Returns
        -------
        seconds_recorded : float

        """
        
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]

        # get file list of csv files with ground truth data 
        filelist = []        
        for folder in folders:
            ver_path = os.path.join(self.main_path, folder, self.verified_dir)
            if os.path.exists(ver_path)== True: # error check
                files = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))
                filelist.extend([os.path.join(ver_path, s) for s in files])
        
        # get total time
        seconds_recorded = 0
        for file in filelist:
            df = pd.read_csv(file)
            seconds_recorded += len(df)
            
        return seconds_recorded
        
    def multi_folder(self):
        """
        Loop though folder paths get seizure metrics and save to csv
    
        Parameters
        ----------
        main_path : Str, group dir name
    
        """
        print('--------------------- START --------------------------')
        print('Testing methods on :', self.main_path)
        
        
        # get subdirectories
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]

        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # add metrics to dataframe
            self.folder_loop(folders[i])
        
        # save dataframe to csv
        self.df['percent_detected'] = 100 *(self.df['detected']/self.df['total'])
        self.df['false_positive_rate'] = self.df['false_positives']/self.duration
        self.df.to_csv(self.output_csv_path, header=True, index=False)
        print('--> Method metrics saved to:', self.output_csv_path ,'\n')
        print('----------------------- END --------------------------')
        

    def folder_loop(self, folder_name):
        """

        Parameters
        ----------
        folder_name : Str, parent dir name

        Returns
        -------
        bool
        """

        # get file list 
        ver_path = os.path.join(self.main_path, folder_name, self.verified_dir)
        if os.path.exists(ver_path)== False: # error check
                print('path not found, skipping:',
                      os.path.join(self.main_path, folder_name) ,'.')
                return False
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
        filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
        
        for file in tqdm(filelist):
            
            # get data and true labels
            data, y_true = get_data(os.path.join(self.main_path, folder_name),          
                                    file, ch_num = ch_list, 
                                    inner_path={'data_path': self.data_dir, 
                                                'pred_path': self.verified_dir},
                                    load_y = True)
            
            # Get features, rearrange data to match df columns, normalize data and get bound of true seizures
            x_data, labels = get_features_allch(data, param_list, cross_ch_param_list)
            x_data = StandardScaler().fit_transform(x_data[:, self.idx])
            bounds_true = find_szr_idx(y_true, dur=self.dur)
            
            for ii in range(len(self.df)): # iterate through df
            
                # detect seizures bigger than threshold
                thresh = (np.mean(x_data) + np.array(self.df.loc[ii][self.thresh]) * np.std(x_data))
                y_pred_array = x_data > thresh
                
                # find predicted seizures
                w = np.array(self.df.loc[ii][self.weights])                     # get weights 
                e =  np.array(self.df.loc[ii][self.enabled])                    # get enabled features
                y_pred = y_pred_array * w * e                                   # get predictions
                y_pred = np.sum(y_pred, axis=1) / np.sum(w * e)                 # normalize to weights and selected features
                y_pred = y_pred > 0.5                                           # get popular vote
                bounds_pred = find_szr_idx(y_pred, dur=1)                       # get bounds of predicted seizures
                
                detected = 0 # set default detected to 0
                if bounds_pred.shape[0] > 0:
                    # number of matching seizures
                    detected = np.sum(match_szrs_idx(bounds_true, y_pred, self.remove_bounds))
                    
                # get total numbers
                self.df.at[ii, 'total'] = self.df['total'][ii] + bounds_true.shape[0]                                       # total true
                self.df.at[ii, 'detected'] = self.df['detected'][ii] + detected                                             # n of matching detected seizures
                self.df.at[ii, 'false_positives'] = self.df['false_positives'][ii] + (bounds_pred.shape[0] - detected)      # n of false positives
                
        return True














