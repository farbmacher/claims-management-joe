# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

import pandas as pd
import json
import numpy as np
import pickle
import sys,os
import ast
import torch
import torch.nn as nn
import multiprocessing

from tqdm import tqdm
from torch.utils.data import Dataset,sampler,DataLoader
from sklearn.model_selection import train_test_split,GroupKFold
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import GroupKFold,KFold
from collections import Counter

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

WODIR=os.getcwd()
DATA_PATH_ORIG=(WODIR + "//data_orig//")
DATA_PATH=(WODIR + "//data_modified//")

CORES_TO_USE=0

def setdif(x,y):
    return list(set(x)-set(y))

def save_as_pickle(name,file):
    with open(name,"wb" ) as fp:
        pickle.dump(file,fp,protocol=pickle.HIGHEST_PROTOCOL)
def load_pickle(name):
    with open(name,"rb" ) as fp:
        return pickle.load(fp)
    
def get_categoricals(frame):
    return [x for x in frame.columns if "category" in x]
def get_numerics(frame):
    return [x for x in frame.columns if "numeric" in x]

def drop_low_category(frame,cati_name,thresh=36):
    vc=frame[cati_name].value_counts()
    frame[cati_name]=frame[cati_name].cat.remove_categories(vc.index[vc<thresh])
    return frame

def get_size_of_folder(folder):
    return sum(os.path.getsize(folder+f) for f in os.listdir(folder) if os.path.isfile(folder+f))/1e9

def blockPrint():
    sys.stdout =open(os.devnull, "w")
def enablePrint():
    sys.stdout=sys.__stdout__
    