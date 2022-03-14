# -*- coding: utf-8 -*-

### ----------------------------- IMPORTS --------------------------- ###
import click
import os
import json
### ----------------------------------------------------------------- ###


def check_main(folder, data_dir, csv_dir):
    """
    Check if folders exist and if h5 files match csv files.

    Parameters
    ----------
    folder : dict, with config settings
    data_dir : str, data directory name
    true_dir : str, csv directory name

    Returns
    -------
    None,str, None if test passes, otherwise a string is returned with the name
    of the folder where the test did not pass

    """
    
    h5_path = os.path.join(folder, data_dir)
    ver_path = os.path.join(folder, csv_dir)
    if not os.path.exists(h5_path):
        return h5_path
    if not os.path.exists(ver_path):
        return ver_path
    h5 = {x.replace('.h5', '') for x in os.listdir(h5_path)}
    ver = {x.replace('.csv', '') for x in os.listdir(ver_path)}   
    if len(h5) != len(h5 & ver):
        return folder

def check_group_dir(settings, data_key='filt_dir', csv_key='true_dir'):
    """
    Check if folders exist and if h5 files in filt directory match csv files.

    Parameters
    ----------
    settings : dict, with config settings
    data_key : str, settings key for filtered data directory
    csv_key : str, settings key for csv directory (can be ground truth or predicted)

    Returns
    -------
    None,str, None if test passes, otherwise a string is returned with the name
    of the folder where the test did not pass

    """
    
    # get child folders and create success list for each folder
    if not os.path.exists(settings['group_path']):
        return settings['group_path']
    folders = [f.path for f in os.scandir(settings['group_path']) if f.is_dir()]
    
    # find whether the same files are present in filtered data and verified files
    for folder in folders:
       check_main(folder, data_dir=settings[data_key], csv_dir=settings[csv_key])

@click.group()
@click.pass_context
def main(ctx):
    """
    -----------------------------------------------------
    
    \b                                                             
    \b                          _                       
    \b                 ___  ___(_)_____   _                            
    \b                 / __|/ _ \ |_  / | | |                                                       
    \b                 \__ \  __/ |/ /| |_| |                                                       
    \b                |___/\___|_/___|\__, |                                                    
    \b                                |___/                                                  
    \b 

    ----------------------------------------------------- 
                                                                                                                                           
    """
        
    # get settings and pass to context
    with open(settings_path, 'r') as file:
        settings = json.loads(file.read())
        ctx.obj = settings.copy()

@main.command()
@click.pass_context
def setgrouppath(ctx):
    """Set path to group folder for processing"""
    
    path = input('Enter Group Path for data processing: \n')
    ctx.obj.update({'group_path': path, 'file_check': False})
    with open(settings_path, 'w') as file:       
        file.write(json.dumps(ctx.obj))  
    click.secho(f"\n -> Group Path was set to:'{path}'.\n", fg='green', bold=True)
    
@main.command()
@click.pass_context
def setmainpath(ctx):
    """Set path to individual folder for verification"""
    
    path = input('Enter Path for seizure verification: \n')
    ctx.obj.update({'main_path': path})
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj))  
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg='green', bold=True)    
        
@main.command()
@click.pass_context
def filecheck(ctx):
    """ Check whether files can be opened and read"""
    from data_preparation.downsample import Lab2h5
    
    # get child folders and create success list for each folder
    if not os.path.exists(ctx.obj['group_path']):
        click.secho(f"\n -> Group folder '{ctx.obj['group_path']}' was not found." +\
                    " Please run -setgrouppath-.\n",
                    fg='yellow', bold=True)
        return
        
    folders = [f.path for f in os.scandir(ctx.obj['group_path']) if f.is_dir()]
    success_list = [] 
    
    for f_path in folders:
        ctx.obj['main_path'] = f_path  
        obj = Lab2h5(ctx.obj)
        success = obj.check_files()
        success_list.append(success)
    
    # save error check to settings file
    ctx.obj.update({'file_check': all(success_list)})
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj)) 
    click.secho(f"\n -> Error check for group folder '{ctx.obj['group_path']}' completed.\n",
                fg='green', bold=True)

@main.command()
@click.option('--p', type=str, help='downsample, filter, predict')
@click.pass_context
def process(ctx, p):
    """Process data (downsample, filter, predict)"""
    
    if not ctx.obj['file_check']:
        click.secho("\n -> File check has not pass. Please run -filecheck-.\n",
                    fg='yellow', bold=True)
        return
    
    process_type_options = ['downsample', 'filter', 'predict']
    if p is None:
        process_type = set(process_type_options)
    else:
        process_type = set([p])
        
    # check if user input exists in process types
    process_type = list(process_type.intersection(process_type_options))
    if not process_type:
        click.secho(f"\n -> Got'{p}' instead of {process_type_options}\n",
                    fg='yellow', bold=True)
        return
         
    # get parent folders (children of group dir)
    folders = [f.path for f in os.scandir(ctx.obj['group_path']) if f.is_dir()]
    
    # process functions
    if 'downsample' in process_type:
        from data_preparation.downsample import Lab2h5
        for f_path in folders:
            ctx.obj['main_path'] = f_path
            Lab2h5(ctx.obj).downsample()
        ctx.obj.update({'downsample':1})
        
    if 'filter' in process_type:
        from data_preparation.preprocess import PreProcess
        for f_path in folders:
            ctx.obj['main_path'] = f_path
            PreProcess(ctx.obj).filter_data()
        ctx.obj.update({'filtered':1})
        
    if 'predict' in process_type:
        from data_preparation.get_predictions import ModelPredict
        for f_path in folders:
            ctx.obj['main_path'] = f_path
            ModelPredict(ctx.obj).predict()
        ctx.obj.update({'predicted':1})
        
    with open(settings_path, 'w') as file:
        file.write(json.dumps(ctx.obj)) 
    return


@main.command()
@click.pass_context
def verify(ctx):
    """Verify detected seizures"""

    out = check_main(folder=ctx.obj['main_path'],
                     data_dir=ctx.obj['filt_dir'],
                     csv_dir=ctx.obj['rawpred_dir'])
    if out:
        click.secho(f"\n -> Main path was not set properly. Could not find: {out}.\n",
             fg='yellow', bold=True)
        return
    
    # import toolbox for verification
    from user_gui.user_verify import UserVerify
    
    # Create instance for UserVerify class
    obj = UserVerify(ctx.obj)
    file_id = obj.select_file()                     # user file selection
    data, idx_bounds = obj.get_bounds(file_id)      # get data and seizure index
    
    # check for zero seizures otherwise proceed with gui creation
    if idx_bounds.shape[0] == 0:
        obj.save_emptyidx(data.shape[0], file_id)     
    else:
        from user_gui.verify_gui import VerifyGui
        VerifyGui(ctx.obj, file_id, data, idx_bounds)
        
        
@main.command()
@click.pass_context
def getprop(ctx):
    """Get seizure properties"""
    
    ver_path = os.path.join(ctx.obj['main_path'], ctx.obj['verpred_dir'])
    if  os.path.exists(ver_path):
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))

    if not filelist:
        click.secho("\n -> Could not find verified seizures: Please verify detected seizures.\n",
             fg='yellow', bold=True)
        return
    
    # get properies and save
    from helper.get_seizure_properties import get_seizure_prop
    _,save_path = get_seizure_prop(ctx.obj)
    click.secho(f"\n -> Properies were saved in '{save_path}'.\n", fg='green', bold=True)
    

@main.command()
@click.option('--p', type=str, help='threshold, parameters, train')
@click.pass_context
def train(ctx, p):
    """Find best parameters"""
    
    # check input
    process_type_options = ['threshold', 'parameters', 'train']
    if p is None:
        process_type = set(process_type_options)
    else:
        process_type = set([p])
    
    # check if user input exists in process types
    process_type = list(process_type.intersection(process_type_options))
    if not process_type:
        click.secho(f"\n -> Got'{p}' instead of {process_type_options}\n",
                    fg='yellow', bold=True)
        return
    
    # get paths from user and check if they are valid
    paths={}
    if 'train' in process_type:
        paths = {'train': 'training data', 'test': 'testing data'}
    elif 'threshold' in process_type:
        paths = {'train': 'training data'}

        
    for i,(key,val) in enumerate(paths.items()):
        path = input('\n' + str(i+1) + '.Enter group path to ' + val + ':\n')
        paths.update({key:path})
        ctx.obj.update({'group_path':path})
        folder = check_group_dir(ctx.obj)
        if folder is not None:
            click.secho(f"\n -> Error in '{folder}'. Could not find .h5 files that match" +\
                        " .csv files in children directories.\n", fg='yellow', bold=True)
            return
    
    if 'threshold' in process_type:
        # find optimum thresholds
        from train.threshold_metrics import ThreshMetrics
        ThreshMetrics(paths['train'], ctx.obj['true_dir']).multi_folder()
        
    if 'parameters' in process_type:
        # create parameter space catalogue
        from train.create_parameter_space import CreateCatalogue
        CreateCatalogue().get_parameter_space()
    
    if 'train' in process_type:
        # get metrics from training and testing datasets
        from train.get_method_metrics import MethodMetrics
        for dataset in paths:
            csv_name = 'parameter_metrics_' + dataset + '.csv'
            MethodMetrics(paths[dataset], ctx.obj['true_dir'],
                          data_dir=ctx.obj['filt_dir'],
                          output_csv_name=csv_name).multi_folder()

        # export best method
        from train.get_best_parameters import get_best
        df,_ = get_best(common_n=1, save=True)
        print_msg = df[['percent_detected', 'false_positive_rate']].to_string()
        click.secho(print_msg, fg='white', bold=True)
        
    if process_type == set(process_type_options):
        click.secho('\n ---> Training was completed successfully.\n', 
                    fg='bright_magenta', bold=True)
       
@main.command()
@click.option('--n', type=str, help='Select number of methods')
@click.option('--s', type=str, help='Select method id')
@click.pass_context
def selbest(ctx, n, s):
    """Select best parameter"""
    
    from train.get_best_parameters import get_best
    import pandas as pd
    
    if not n:
        n = 1
    else:
        n = int(n)
        
    # select best method
    df, save_path = get_best(common_n=n)
    print_msg = df[['percent_detected', 'false_positive_rate']].to_string()
    click.secho('\n' + print_msg + '\n', fg='white', bold=True)
    
    if not s:
        s = df.index[0]
    else:
        s = int(s)
    
    # save dataframe
    df = pd.DataFrame(df.loc[s])
    df.T.to_csv(save_path, index=False)
    print_msg = '--> Index: ' + str(s) +\
        ' Best parameter-set was exported to: ' + save_path + '\n'
    click.secho(print_msg, fg='white', bold=True)
    

if __name__ == '__main__':
    
    # define settings path
    temp_settings_path = 'temp_config.json'
    settings_path = 'config.json'
    
    # check if settings file exist and if all the fields are present
    if not os.path.isfile(settings_path):
        import shutil
        shutil.copy(temp_settings_path, settings_path)
        
    else:
        # check if keys match otherwise load original settings
        with open(temp_settings_path, 'r') as file:
            temp_settings = json.loads(file.read())          
        with open(settings_path, 'r') as file:
            settings = json.loads(file.read())
    
        if settings.keys() != temp_settings.keys():
            import shutil
            shutil.copy(temp_settings_path, settings_path)
        
    # init cli
    main(obj={})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    