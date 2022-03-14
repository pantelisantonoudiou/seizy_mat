# -*- coding: utf-8 -*-


### ------------------------------ IMPORTS ------------------------------- ###
import os
import numpy as np
import pandas as pd
from functools import reduce
### ---------------------------------------------------------------------- ###

def get_best(csv_dir='train', common_n=10, best_n=100, save=False,
                    output_csv_name='selected_method.csv',
                    files = ['parameter_metrics_train.csv',
                             'parameter_metrics_test.csv']):
    
    # get cost and compare dataframes
    idx = []
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(csv_dir, file))
        df['cost'] = df['false_positive_rate'] - df['percent_detected']
        idx.append(set(df.nsmallest(best_n, 'cost').index.values))
    
    # get common best methods
    common_idx = np.array(list(reduce(set.intersection, map(set, [idx[0], idx[1]]))))
    
    if common_n > len(common_idx):
        common_n = len(common_idx)
        
    # trim to user specified length
    df_best = df.loc[common_idx[:common_n]]
    
    # save best selected method
    file_path = os.path.join(csv_dir, output_csv_name)
    
    if save:
        df_best.to_csv(file_path, index=False)
        print('--> Best parameter-set was exported to:', file_path, '\n' )

    return df_best, file_path

if __name__ == '__main__':
    df,_ = get_best(csv_dir='', common_n=10, best_n=20,)