# -*- coding: utf-8 -*-

### ------------------------------ IMPORTS ------------------------------ ###
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from train.feature_settings import param_list, cross_ch_param_list
from train.feature_settings import metrics, feature_labels, ch_list
from helper.io_getfeatures import get_data, get_features_allch
from helper.array_helper import find_szr_idx, match_szrs_idx
### --------------------------------------------------------------------- ###

class ThreshMetrics:
    """
    Get metrics across thresholds for each parameter in feature_labels.
    """
    
    def __init__(self, main_path, verified_dir, data_dir='filt_data',
                 csv_dir='train', output_csv_name='threshold_metrics.csv'):
        """

        Parameters
        ----------
        main_path : str, path to parent directory.
        verified_dir : str, name of verified folder with binary csv files.
        data_dir : str, name of folder with h5 voltage data.
        csv_dir : str, directory to save output save.
        output_csv_name : str, name of output csv file.
        
        Returns
        -------
        None.
        
        --- Examples ---
        ThreshMetrics('\data\training_data', 'verified_predictions').multi_folder()
        """

        # pass parameters to object
        self.main_path = main_path
        self.verified_dir = verified_dir
        self.data_dir = data_dir
        self.output_csv_path = os.path.join(csv_dir, output_csv_name)
        
        # get paremeters from module
        self.ch_list = ch_list
        self.metrics = metrics
        self.feature_labels = feature_labels
        
        # get total time in hours
        self.duration = self.get_total_duration()/3600
        
        # define bounds to remove short seizure segments
        self.dur = 1
        self.remove_bounds = np.array([0,1])
        
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
        
        Returns
        -------
        save_df : pd.DataFrame, containing threshold metrics
    
        """
        print('--------------------- START --------------------------')
        print('------------------------------------------------------')
        print('Getting metrics from :', self.main_path)

                
        # get threhsolds
        threshold_array = np.linspace(2,6,9);
        
        save_df = pd.DataFrame()
        for ii in range(threshold_array.shape[0]): # iterate though thresholds
            
            self.threshold = threshold_array[ii] # set threshold
            print('Detection threshold set at :', self.threshold,'\n')
        
            # create df for storage of metrics
            self.df = pd.DataFrame(
                data = np.zeros((len(self.feature_labels), 
                                len(self.metrics))), 
                                columns = self.metrics)
            self.df.insert(loc = 0, column ='features', value = self.feature_labels)
                
            # get subdirectories
            folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
        
            for i in range(len(folders)): # iterate through folders
                print('Analyzing', folders[i], '...' )
                
                # add metrics to dataframe
                self.folder_loop(folders[i])
            
            # add threshold to df and concatenate
            self.df['threshold'] = self.threshold      
            save_df = pd.concat((save_df, self.df), axis=0)
            
        # calculate a few more metrics and save dataframe to csv
        save_df['percent_detected'] = 100 *(save_df['detected']/save_df['total'])
        save_df['false_positive_rate'] = save_df['false_positives']/self.duration
        save_df.to_csv(self.output_csv_path, header=True, index=False)
        print('Seizure metrics saved to:', self.output_csv_path, '\n')
        print('----------------------- END --------------------------')
        print('------------------------------------------------------')
        return save_df

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
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))
        filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
     
        for file in tqdm(filelist): # iterate through experiments
    
            # get data and true labels
            data, y_true = get_data(os.path.join(self.main_path, folder_name),
                                    file, ch_num = self.ch_list, 
                                    inner_path={'data_path':self.data_dir, 
                                    'pred_path':self.verified_dir}, 
                                    load_y = True)
            
            # Get features, normalize data and get bound of true seizures
            x_data, labels = get_features_allch(data, param_list, cross_ch_param_list)
            x_data = StandardScaler().fit_transform(x_data)
            bounds_true = find_szr_idx(y_true, dur=self.dur)
            
            if bounds_true.shape[0] > 0:  # proceed if seizures are present  
            
                for ii in range(len(self.feature_labels)): # iterate through parameters
        
                    # get bounds of predicted sezures
                    y_pred = x_data[:,ii]> (np.mean(x_data[:,ii]) + self.threshold*np.std(x_data[:,ii]))
                    bounds_pred = find_szr_idx(y_pred, dur=1)                                   # total predicted              
                    detected = np.sum(match_szrs_idx(bounds_true, 
                                                     y_pred, self.remove_bounds))         # find matching seizures

                    # get total numbers
                    self.df.at[ii, 'total'] += bounds_true.shape[0] 
                    self.df.at[ii, 'detected'] += detected
                    self.df.at[ii, 'false_positives'] += (bounds_pred.shape[0] - detected)
        return True

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


