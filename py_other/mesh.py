import numpy as np

def mesh(x1,x2):
    """
    Formated mesh of vectors x1, x2
    Output is numpy array [p,2], congruent with 'X' matrix used for positions in Localization
    functions and classes
    """
    x1v, x2v = np.meshgrid(x1,x2)
    x1v = np.reshape(x1v,(np.prod(x1v.shape),1))
    x2v = np.reshape(x2v,(np.prod(x2v.shape),1))
    XM  = np.concatenate((x1v,x2v),axis=1)
    return XM
