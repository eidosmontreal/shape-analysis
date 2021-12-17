import matplotlib
matplotlib.use('Agg') # Comment out if not ssh'ing in
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from os.path import join
import os, json

from typing import List, NoReturn

cmap = plt.get_cmap('viridis')
"""
This file contains plotting functions for training and testing data generated during model training.
The functions heavily rely on the naming conventions found in the various training scripts in`train/`.
As such, they should primarily be used in tandem with the said training scripts.
"""

def moving_average(x: np.array,window_size: int=5) -> np.array:
    """
    Compute moving average of input array.

    :param x: np.array
    :param window_size: Size of window to take average upon
    """
    y = np.zeros(x.shape)
    y[:window_size] = x[:window_size]
    for i in range(window_size,len(x)):
        y[i] = np.mean(x[i-window_size:i])
    return y

def plot_all_iterations(log_dir:str,csv_name: str='train_iterations.csv',fname: str='training_plot.png',init_epoch:int=0,last_epoch: int=-1) -> NoReturn:
    """
    Returns plot of all iterations over all epochs in ``csv_name`` and outputs it to ``fname``.

    :param log_dir: Directory containing .csv
    :param csv_name: Name of .csv to be read
    :param fname: Name of output file
    :param init_epoch: Epoch to start plot from
    :param last_epoch: Final epoch to stop plotting
    """
    df = pd.read_csv(os.path.join(log_dir,csv_name))
    if last_epoch == -1:
        last_epoch = df['epoch'].max()
    df = df[df['epoch']>=init_epoch]
    df = df[df['epoch']<=last_epoch]
    
    errors = df.groupby('epoch').std().to_numpy().transpose(1,0)
    data = df.groupby('epoch').mean().to_numpy().transpose(1,0)
    epochs = range(init_epoch,last_epoch+1)
    
    color = cmap(np.random.uniform(0,1))
    plt.plot(epochs,data[1],c=color)[0]
    plt.fill_between(epochs,data[1]-errors[1],data[1]+errors[1],
                alpha = 0.2,
                color=color,
                linewidth=2)
    plt.grid(True)
    plt.savefig(os.path.join(log_dir,fname))
    plt.close()

def plot_multiple_directories(log_dirs: List[str],csv_name: str='train.csv',col: str='loss',fname: str='summary.png',window_size: int=5,labels: List[str]=None,title: str='',init_epoch: int=0,last_epoch: int=-1) -> NoReturn:
    """
    Returns plot of all epochs in ``csv_name`` over several directories and outputs it to ``fname`.

    :param log_dirs: List of directories containing .csv
    :param csv_name: Name of .csv to be read from each directory
    :param col: WHich column to plot in .csv
    :param fname: Name of output file
    :param window_size: Window size of moving average that is applied to ``col``
    :param labels: Optional list of labels for each directory used in legend
    :param title: Title of output plot
    :param init_epoch: Epoch to start plot from
    :param last_epoch: Final epoch to stop plotting
    """
    if labels is None:
        no_labels=True
        labels = []
    else:
        no_labels=False
    for idx,log_dir in enumerate(log_dirs):
        with open(join(log_dir,'args.json'),'r') as f:
            args = json.load(f)
        df = pd.read_csv(os.path.join(log_dir,csv_name))
        df = df[df['epoch']>=init_epoch]
        if last_epoch != -1:
            df = df[df['epoch']<last_epoch]
        if idx == 0:
            df_main = df.copy()
        data = moving_average(df[col]) #Remove any unwanted nans
        df_main[col+str(idx)] = data
        if no_labels:
            labels += [args['dir_name']]
    df_main.plot(x='epoch',y=[col+str(i) for i in range(0,len(log_dirs))],grid=True)
    plt.legend(labels)
    plt.ylabel(col + ' (moving avg. over {} epochs)'.format(window_size))
    plt.title(title)
    plt.savefig(fname)
    plt.close()

def plot_train_test(log_dirs: List[str],fname: str='summary.png',window_size: int=5,title: str='',init_epoch: int=0,last_epoch: int=-1) -> NoReturn:
    """
    Returns array of plots of train and test data for each respective directories. Assumes that the data is kept in `train_test.csv`, as per the scripts in ``train/``, with column labels `loss` and `accuracy`.

    :param log_dirs: List of directories containing .csv
    :param fname: Name of output file
    :param window_size: Window size of moving average that is applied to ``col``
    :param title: Title of output plot
    :param init_epoch: Epoch to start plot from
    :param last_epoch: Final epoch to stop plotting
    """
    labels = []
    fig,axes = plt.subplots(len(log_dirs))
    if len(log_dirs) == 1:
        axes = [axes]
    for ax,log_dir in zip(axes,log_dirs):
        
        with open(join(log_dir,'args.json'),'r') as f:
            args = json.load(f)
        
        df = pd.read_csv(os.path.join(log_dir,'train_test.csv'))

        df = df[df['epoch']>=init_epoch]
        if last_epoch != -1:
            df = df[df['epoch']<last_epoch]
        train_plot = ax.plot(moving_average(df['loss'],window_size=window_size),color='red',label='Train')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss (averaged)')
        plt.grid()

        ax2 = ax.twinx()
        test_plot = ax2.plot(moving_average(df['accuracy'],window_size=window_size),color='blue',label='Test')
        ax2.set_ylabel('Test accuracy (averaged)')
       
        ax.legend(train_plot+test_plot,['Train','Test'],loc='best')
        plt.title('{} trained on {}'.format(args['model'],args['dataset']))
    fig.suptitle(title) 
    plt.savefig(os.path.join(log_dir,fname))
    plt.close()
