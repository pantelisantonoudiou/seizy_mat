# -*- coding: utf-8 -*-

### ------------------------- IMPORTS ------------------------ ###
import os
import numpy as np
import pandas as pd
from train.feature_settings import metrics
### ---------------------------------------------------------- ###

class CreateCatalogue:
    """
    Create parameter catalogue from best thresholds for model training.
    """
    
    def __init__(self, csv_dir='train', metrics_csv='threshold_metrics.csv',
                 output_csv_name='parameter_catalogue.csv'):
        """

        Parameters
        ----------
        csv_dir : str, directory to save output save.
        metrics_csv : str, name of metrics csv file.
        output_csv_name : str, name of output csv file.

        Returns
        -------
        None.

        """
        
        # get paths
        self.metircs_csv_path = os.path.join(csv_dir, metrics_csv)
        self.output_csv_path = os.path.join(csv_dir, output_csv_name)
        
        # get thresholds and feature labels
        self.thresholds = self.get_thresholds()
        self.feature_labels = self.thresholds.columns[1:]
        
        # get feature parameters for method testing
        self.thresh_array, self.weights, self.feature_set = self.get_feature_parameters()
        
        # define metrics
        self.metrics = metrics

    def get_thresholds(self):
        """
        Get best thresholds and ranks.
    
        Returns
        -------
        thresholds : pd.DataFrame
    
        """
        
        # load metrics
        df = pd.read_csv(self.metircs_csv_path)
        
        # find threshold for minimum cost
        df['cost'] = df['false_positive_rate'] - df['percent_detected']
        min_cost = df.loc[df.groupby('features').cost.idxmin()]
        
        # combine thresholds with ranks
        thresholds = pd.DataFrame(min_cost[['threshold', 'features']])
        thresholds['weights'] = len(min_cost['features']) - min_cost['cost'].rank()
        thresholds['cost'] = min_cost['cost']
        
        # format dataframe
        thresholds = thresholds.T
        column_name = 'features'
        thresholds.columns = thresholds.loc[column_name]
        thresholds = thresholds.drop(thresholds.index[1])
        thresholds = thresholds.rename_axis('metrics').reset_index()
        
        return thresholds


    def get_feature_parameters(self, n_repeat=500):
        """
        Get feature parameter combinations for method testing.
        
        Parameters
        ----------
        n_repeat : int, number of times to add random features per dataset.
        
        Returns
        -------
        thresh_array : list
        weights : list
        feature_set : list
    
        """
       
        
        # get feature properties
        df = self.thresholds
        features = np.array(self.feature_labels).reshape(1,-1)
        ranks = np.array(df.loc[df['metrics'] == 'weights'])[0][1:]
        ranks = ranks.astype(np.double)
        optimum_threshold = np.array(df.loc[df['metrics'] == 'threshold'])[0][1:]
        optimum_threshold = optimum_threshold.astype(np.double)
        
        # define different threshold levels for testing
        thresh_array = []
        add_to_optimum_thresh = np.arange(-1, 2.5, .5)
        add_to_thresh = np.arange(2, 4, .5)
        for opt_threshold, reg_threshold  in zip(add_to_optimum_thresh, add_to_thresh):
            thresh_array.append(optimum_threshold + opt_threshold)
            thresh_array.append(np.ones((optimum_threshold.shape[0])) * reg_threshold)
        
        # define two sets of weights
        weights = [np.ones((features.shape[1])), ranks]
        
        # define feature sets
        feature_set_or = [np.ones((ranks.shape[0]), dtype=bool),
                          ranks > np.percentile(ranks, 50), 
                          ranks > np.percentile(ranks, 75)]
        n_repeats = n_repeat * np.array([0.01, 0.8, 0.1])
        n_repeats = n_repeats.astype(int)
        
        # expand feature dataset by randomly dropping selected features
        feature_set = feature_set_or.copy()
        for i in range(len(feature_set_or)):                                   # iterate through original dataset
            len_temp = sum(feature_set_or[i])
            max_drop = int(len_temp - 2)
            min_drop = int(len_temp/2)
            for ii in range(n_repeats[i]):                                     # iterate n times to drop random features
                temp_feature = feature_set_or[i].copy()             
                drop_n = np.random.randint(min_drop, max_drop)
                true_idx = np.where(temp_feature)[0]
                idx = np.random.choice(true_idx, drop_n, 
                                       replace=False)
                temp_feature[idx] = False
                feature_set.append(temp_feature)
        
        # get unique feature combinations
        feature_set = [np.array(x) for x in set(tuple(x) for x in feature_set)]
    
        return thresh_array, weights, feature_set

    def get_parameter_space(self):
        """
        Create self dataframe  based on thresholds, weighs and feature set.

        Returns
        -------
        df : pandas DataFrame

        """

        # get df columns
        columns = self.metrics + ['Thresh_' + x for x in self.feature_labels] \
        + ['Weight_' + x for x in self.feature_labels] + ['Enabled_' + x for x in self.feature_labels]
        
        # create df 
        rows = len(self.thresh_array) * len(self.weights) *len(self.feature_set)
        df = pd.DataFrame(data= np.zeros((rows, len(columns))), columns = columns)
        
        # get index
        idx2 = len(self.metrics) + len(self.feature_labels)
        idx3 = idx2 + len(self.feature_labels)
        
        cntr = 0; # init cntr
        for thresh in self.thresh_array:
            for weight in self.weights:
                for feature in self.feature_set:
                    df.loc[cntr][len(self.metrics):idx2] = thresh
                    df.loc[cntr][idx2:idx3] = weight
                    df.loc[cntr][idx3:] = feature.astype(np.double)
                    cntr+=1 # update counter
                    
        df.to_csv(self.output_csv_path, index=False)
        print('--> Parameter catalogue stored in:', self.output_csv_path, '\n')
        return df


if __name__ =='__main__':
    
    # get parameter space catalogue
    df_catalogue = CreateCatalogue().get_parameter_space()
    # df_catalogue.to_csv('template_catalogue.csv', index=False)
    
    
    
    
    
    
    



