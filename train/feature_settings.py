# -*- coding: utf-8 -*-

### -------------- IMPORTS ------------- ###
import numpy as np
from helper import features
### ------------------------------------ ###

# define metrics
metrics = ['total', 'detected', 'percent_detected',
                        'false_positives', 'false_positive_rate']

## define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, 
              features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,)

# create feature labels
ch_list = [0, 1] # define channel_list
feature_labels = []
for n in ch_list:
    feature_labels += [x.__name__ + '_'+ str(n) for x in param_list]
feature_labels += [x.__name__  for x in cross_ch_param_list]
feature_labels = np.array(feature_labels)


