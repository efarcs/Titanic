import pandas as pd
import numpy as np
import regex as re
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns



#Titanic function for imputating cabins 
def cabin_imputation(df, pclass: str):

    # non-null cabins for this class into a weighted distribution
    notations = df.loc[df['Cabin'].notna() & (df['Pclass'] == pclass), 'Cabin'].str[0]
    print(notations)
    freqs = notations.value_counts(normalize=True)
    print(freqs)

    #Creating a mask for null cabins and then imputating 
    mask = df['Cabin'].isna() & (df['Pclass'] == pclass)
    n = mask.sum()
    df.loc[mask, 'Cabin_Imputation_Flag'] = True
    df.loc[mask, 'Cabin_Notation'] = np.random.choice(freqs.index, size=n, p=freqs.values)