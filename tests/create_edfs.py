# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pyedflib
##### ------------------------------------------------------------------- #####

class EdfMaker:
    """
    Create EDFs to test seizure detection.
    """
    
    def __init__(self, properties):
        
        # get values from dictionary
        for key, value in properties.items():
               setattr(self, key, value)
        
        # get channel numbers in python format
        self.animal_ch_list = np.array(self.animal_ch_list) - 1
        
        # create time vector
        self.t = np.arange(0, self.time_duration, 1/self.fs)
        
        # get info for channels
        self.channel_info = []

        for ch_list in self.animal_ch_list:
            for i, ch in enumerate(ch_list):
                # get channel properties
                ch_dict = {'label': self.ch_id[i], 'dimension': self.dimension[i], 
                   'sample_rate': self.fs, 'physical_max': self.physical_max[i],
                   'physical_min': self.physical_min[i], 'digital_max': self.digital_max[i], 
                   'digital_min': self.digital_min[i], 'transducer': '', 'prefilter':''}
                
                # append info
                self.channel_info.append(ch_dict)

    
    @staticmethod
    def make_sine(t, freq:float, amp:float):
        """
        Create sine wave.

        Parameters
        ----------
        t : array, time
        freq : float, frequency
        amp : float, amplitude

        Returns
        -------
        array, sine wave

        """
        return np.sin(freq*t*np.pi*2) * amp
 
           
    def create_data(self):
        """
        Create data for all animals in a file.

        Returns
        -------
        data: list, containing arrays for each edf channel.

        """
        
        data = []
        
        # iterate over animal lists
        for ch_list, szr_prop in zip(self.animal_ch_list, self.seizure_time_len):
            
            # iterate over channels for each animal
            for ch in ch_list:
                # append data for each channel
                data_ch = self.create_data_one(szr_prop)
                data.append(data_ch * self.scale)  
                
        return data

        
    def create_data_one(self, szr_prop):
        """
        Create sine waves at specified locations for one channel.

        Parameters
        ----------
        szr_prop : dict, {seizure time occurence(s): seizure duratiion(s)}

        Returns
        -------
        data_ch : array, 

        """
        
        
        data_ch = np.random.rand(self.t.size)
        for szr_time, szr_dur in szr_prop.items():
            
            # get index
            x1 = szr_time*self.fs
            x2 = (szr_time + szr_dur)*self.fs
            
            # create sine wave
            t = np.arange(x1, x2, 1)/ self.fs
            wave = self.make_sine(t, self.freq, self.amp)
            data_ch[x1:x2] = wave
            
        return data_ch
    
    
    def create_edf(self):
        """
        Create_edf file.
        
        Parameters
        ----------
        block : int, labchart block number
        
        Returns
        -------
        None.
        """
        

        # intialize EDF object
        file_name = '_'.join(obj.animal_ids)
        file_path = os.path.join(self.save_path, file_name+ '.edf')
        
        with pyedflib.EdfWriter(file_path, self.animal_ch_list.size,
                                file_type = pyedflib.FILETYPE_EDF) as edf:
            
            # write headers
            edf.setSignalHeaders(self.channel_info)
                
            # write data to edf file     
            data = self.create_data()
            edf.writeSamples(data)
    
            # close edf file writer
            edf.close()
        print('Edf was created:', file_path)
    
    

if __name__ == '__main__':
    
    properties = {"save_path": r"",
        "fs" : 1000,
        "time_duration": 18*60*60,
        "ch_id":  ["lfp","eeg","emg"],
        "dimension": ["V","V","V"],
        "physical_max": [0.1, 0.1, 0.01],
        "physical_min": [-0.1, -0.1, -0.01],
        "digital_max": [32000, 32000, 32000],
        "digital_min": [-32000, -32000, -32000],
        "scale": 1e-3,
        "animal_ch_list":  [[1,2,3], [4,5,6], [7,8,9], [10,11,12]],
        "animal_ids":  ['1001', 'mouse2', '4506', 'mouse4'],
        "amp": 2,
        "freq": 8,
        "seizure_time_len": [{100*60:50, 200*60:70, 520*60:66}, {130*60:20, 230*60:30, 550*60:40},
                             {100*60:70, 200*60:90, 502*60:80}, {250*60:20, 400*60:72, 502*60:63}],
        }
    
    obj = EdfMaker(properties)
    obj.create_edf()
    
    data = obj.create_data()
    
    # import matplotlib.pyplot as plt
    # from matplotlib.pyplot import cm
    
    # color = cm.Paired(np.linspace(0, 1, len(data)))
    # f,axs = plt.subplots(len(data),1)
    
    # for i, (ax, data_ch) in enumerate(zip(axs, data)):
    #     ax.plot(obj.t, data_ch, c=color[i,:])
            
            
            
    

        
            
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            