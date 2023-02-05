from ..cunnData import cunnData

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def scatter(cudata:cunnData, 
                x:str, 
                y:str,
                color:str = None,
                save:str = None,
                show:bool =True,
                dpi:int =300)-> None:
    """
    Violin plot.
    Wraps :func:`seaborn.scatterplot` for :class:`~rapids_singlecell.cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form :func:`scanpy.pl.scatter`.
    
    Parameters
    ---------
    cudata
        cunnData object
    x
        Keys for accessing variables of fields of `.obs`.
    y
        Keys for accessing variables of fields of `.obs`.
    save
        file name to save plot as in ./figures
    show
        if you want to display the plot
    dpi
        The resolution in dots per inch for save

    
    """
    fig,ax = plt.subplots()
    if color == None:
        sns.scatterplot(data=cudata.obs, x=x, y=y,s=2, color="grey", edgecolor="grey")
    else:
        sns.scatterplot(data=cudata.obs, x=x, y=y,s=2, hue=color)

    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()

        
def violin(cudata:cunnData,
            key:str,
            groupby:str=None,
            size:float =1,
            save:str = None,
            show:bool =True,
            dpi:int =300):
    """
    Violin plot.
    Wraps :func:`seaborn.violinplot` for :class:`~rapids_singlecell.cunnData.cunnData`. This plotting function so far is really basic and doesnt include all the features form :func:`scanpy.pl.violin`.
    
    Parameters
    ---------
        cudata
            cunnData object
        key
            Keys for accessing variables of fields of `.obs`.
        groupby
            The key of the observation grouping to consider.(e.g batches)
        size
            pt_size for stripplot if 0 no strip plot will be shown.
        save
            file name to save plot as in ./figures
        show
            if you want to display the plot
        dpi
            The resolution in dots per inch for save
    
    Returns
    ------
    nothing
    
    """
    fig,ax = plt.subplots()
    ax = sns.violinplot(data=cudata.obs, y=key,scale='width',x= groupby, inner = None)
    if size:
        ax = sns.stripplot(data=cudata.obs, y=key,x= groupby, color='k', size= size, dodge = True, jitter = True)
    if save:
        os.makedirs("./figures/",exist_ok=True)
        fig_path = "./figures/"+save
        plt.savefig(fig_path, dpi=dpi ,bbox_inches = 'tight')
    if show is False:
        plt.close()
