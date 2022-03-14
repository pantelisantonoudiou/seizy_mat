# -*- coding: utf-8 -*-

### ------------------------ IMPORTS -------------------------------------- ###
import os
import json
import tables
import adi
from tqdm import tqdm
import numpy as np
from scipy import signal
from string import ascii_lowercase
from helper.array_helper import chunk_array
### ------------------------------------------------------------------------###


class Lab2h5:
    """
    Class for convering labchart files to H5 format.
    
    Description
    -----------
    One labchart file may contain recordings from multiple animals.
    Each animal may have multiple channels.
    For example one file-> 1-12 channels and 4 animals.
    One H5 file will contain all channels for 1 animal (eg. 1-3).
    
    Assumptions
    -----------
    1) Parent folder contains animal IDs separated by underscore in recording order.
    2) The animals and their position is consistent across sessions.
    3) Channels that belong to one animal have the same sampling rate.
    
    """
   
    # class constructor (data retrieval)
    def __init__(self, property_dict):
        """
        lab2mat(main_path)

        Parameters
        ----------
        property_dict : Dict contianing essential parameters for conversion

        """
        # declare instance properties
        self.ch_struct = property_dict['ch_struct']                             # channel structure
        self.win = property_dict['win']                                         # window size in seconds
        self.animal_ids = []                                                    # animal IDs from folder
        self.load_path =''                                                      # full load path
        self.save_path = ''                                                     # full save path
        self.org_dir = property_dict['org_dir']                                 # reorganized data folder
        self.file_ext = '.adicht'                                               # file extension
        self.chunksize = property_dict['chunksize']                             # number of rows to be read into memory
        self.new_fs = property_dict['new_fs']                                   # new sampling rate
        self.cols =  self.win * self.new_fs                                     # get window size in samples
        self.ch_select = property_dict['ch_list']                               # channels to analyze per animal
        self.chunkshape = [self.chunksize, self.cols, len(self.ch_select)]      # shape of data chunks for file read
        self.ds_shape = [0, self.cols, len(self.ch_select)]                     # shape of ds table array
        self.properties = property_dict
        
        # get animal ids and load path
        self.animal_ids = property_dict['main_path'].split(os.sep)[-1].split('_')
        self.gen_path = property_dict['main_path']
        self.load_path = os.path.join(self.gen_path, property_dict['data_dir'])

        # get save path and file list
        self.save_path = os.path.join(self.gen_path, self.org_dir)
        self.filelist = list(filter(lambda k: self.file_ext in k, os.listdir(self.load_path)))


    def downsample(self):
        """
        Iterate over files in parent folder.

        """
        print('---------------------------------------------------------------------------\n')
        print('---> Initiating File Conversion for:', self.load_path + '->',
              len(self.filelist), 'Labchart files will be analyzed.\n')
        
        # make path and save config
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
            open(os.path.join(self.gen_path,'properties.json'), 'w').write(json.dumps(self.properties))
        
        self.cntr = 0 # init counter
        
        # loop through labchart files (multilple animals per file)
        for i in range(len(self.filelist)):
            
            # get file read object and channel list
            f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
            ch_idx = np.linspace(1, f.n_channels, f.n_channels, dtype=int)
            ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))

            for ii in range(len(ch_list)): # iterate through animals
                
                # downsample and save in chuncks
                filename = self.filelist[i].replace(self.file_ext, "") + '_' + self.animal_ids[ii]
                self.save_chunks(f, filename, ch_list[ii][self.ch_select])

        print('\n--->  File Conversion Completed.', self.cntr, '\n Files Were Saved To:', self.save_path+'.', '\n')
        print('---------------------------------------------------------------------------\n')
    
    
    def save_chunks(self, file_obj, filename, ch_list):
        """
        Decimate and save data to h5 files in chunnks for one file.

        Parameters
        ----------
        file_obj : adi file object,
        filename : str,
        ch_list : array, containing channels for each animal.
            e.g. [1,2,3], [4,5,6]...

        Returns
        -------
        None.

        """
        
        # convert channel to python format and get all blocks
        ch_list = ch_list - 1                                  
        all_blocks = len(file_obj.channels[0].n_samples)
        
        for self.block in range(all_blocks):
            
            print(self.cntr+1,'-> Converting block :', self.block, 'in File:', filename)
            
            # skip corrupted blocks
            try:
                chobj = file_obj.channels[ch_list[0]] # get first channel
                chobj.get_data(self.block+1, start_sample=1, stop_sample=1000)
                
            except:
                print('Block :', self.block, 'in File:', filename, 'is corrupted.')
                continue

            # create data store to save each block
            file_id  = filename + ascii_lowercase[self.block] + '.h5'
            full_path = os.path.join(self.save_path, file_id)
            fsave = tables.open_file(full_path, mode='w') 
            ds = fsave.create_earray(fsave.root, 'data', tables.Float64Atom(),
                                     self.ds_shape, chunkshape=self.chunkshape)
            
            # decimate data and append to datastore
            self.append_to_ds(file_obj, ds, ch_list)
            fsave.close()
            self.cntr += 1


    def append_to_ds(self, file_obj, ds, ch_list):
        """
        Append decimated data to datastore.

        Parameters
        ----------
        file_obj : adi file object,
        ds : pytables datastore
        ch_list : array, number of channels corresponding to current animal
        Returns
        -------
        None.

        """
        
        # split file size to chuncks
        fs = file_obj.channels[ch_list[0]].fs[self.block]                   # get sampling rate
        win_size = int(self.win*fs)                                         # get window size
        self.down_factor = round(fs/self.new_fs)                            # get downsampling factor
        length = file_obj.channels[ch_list[0]].n_samples[self.block]        # get file length
        length -= length % win_size                                         # trim to window size
        idx = chunk_array(1, length, self.chunksize*win_size)
        
        # Iterate over chunks
        for i in tqdm(range(len(idx)), desc = 'Progress'):
            
            # get decimated data and append to datastore
            data = self.get_filechunks(file_obj, idx[i], ch_list)
            ds.append(data)
 

    def get_filechunks(self, file_obj, index, ch_list):
        """
        Segment labchart file to numpy array.

        Parameters
        ----------
        file_obj : adi file object,
        index : two element vector int
            start and stop index in window blocks.

        Returns
        -------
        data : numpy array

        """
        
        # get rows and create empty data with appropriate shape
        rows = int((index[1]-index[0]+1)/self.down_factor/self.cols)
        data = np.zeros((rows, self.cols, 0))

        # iterate over channels and concatenate data
        for i in range(len(ch_list)):      
            
            # retrieve and decimate data
            chobj = file_obj.channels[ch_list[i]]
            data_ch = chobj.get_data(self.block+1, start_sample=index[0], stop_sample=index[1])
            data_ch = data_ch.astype(np.float64)
            data_ch = signal.decimate(data_ch, self.down_factor)
            
            # reshape and concatenate
            data_ch = data_ch.reshape((rows, self.cols, 1))
            data = np.concatenate((data, data_ch), axis=2)

        return data


    ### FILE CHECK ####
    def check_files(self):
        """
        Check if files in parent folder can be read or skipped successfully.
        
        Returns
        -------
        bool, True if operation succesfull 
    
        """
        
        print('--> Initiating Error Check for', self.load_path)
        print('---> Step 1 : Testing file opening ... \n')
        
        self.cntr = 0 # init file counter
        
        # loop through labchart files (multilple animals per file)
        for i in  range(len(self.filelist)):
            
            # get adi file obj
            f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
            
            # get channel list     
            ch_idx = np.linspace(1,f.n_channels,f.n_channels,dtype=int)
            ch_list = []; # init empty channel list
            
            # check if chanels are divisible by channel structure
            if f.n_channels % len(self.ch_struct) == 0:

                # split according to channel length
                ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))
            
            # check if channel list length matches length of animals
            if len(ch_list) - len(self.animal_ids) != 0:
                print('********** Error!!! Animal numbers do not match channel structure!!! **********\n')
                return False

            for ii in range(len(ch_list)): # iterate through animals
                
                # test if blocks in file can be read or skipped
                filename = self.filelist[i].replace(self.file_ext, "") + '_' + self.animal_ids[ii]
                self.test_files(f,filename,ch_list[ii])
        
        print('\n--- >', self.cntr, 'files were opened or skipped successfully.\n')
        print('--->  Error Check for', self.load_path, 'completed. \n')
        print('---------------------------------------------------------------------------\n')
 
        return True


    def test_files(self, file_obj, filename, ch_list,
                   nsamples=1000):
        """
        Tries to read start, middle and end of all blocks in a file.

        Parameters
        ----------
        file_obj : adi file object,
        filename : str,
        ch_list : array, Containing channels for each animal.
            e.g. [1,2,3], [4,5,6]...
        nsamples: int, number of samples to read from each file portion.
        
        Returns
        -------
        None.

        """
        
        # convert channel to python format and get all blocks
        ch_list = ch_list - 1 
        all_blocks = len(file_obj.channels[0].n_samples)
        
        for block in range(all_blocks):

            print(self.cntr+1,'-> Reading from block :', block, 'in File:', filename)
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            length = chobj.n_samples[block] # get block length in samples
            
            try: # skip corrupted blocks
                chobj.get_data(block+1, start_sample=1,                         # start
                               stop_sample=nsamples)   
                chobj.get_data(block+1, start_sample=int(length/2),             # middle
                               stop_sample=int(length/2)+nsamples) 
                chobj.get_data(block+1, start_sample=length-nsamples,           # end
                               stop_sample=length-1) 
                self.cntr+=1
            except:
                print('Block :', block, 'in File:', filename, 'is corrupted')
                continue


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        