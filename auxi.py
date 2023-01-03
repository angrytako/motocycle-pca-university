import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
def countElems(seriesNoNa):
    count = defaultdict(int)
    for name in seriesNoNa:
        count[name] +=1
    return sorted(count.items(), key=lambda x: x[1], reverse=True)
    
def substitute_with_mean(series):
    mean = series.dropna().mean()
    std = series.dropna().std()
    new_val = np.random.normal(loc=mean, scale=std)
    return series.interpolate(method='linear').ffill().bfill()

def plot_densities(df, features, shape, title):
    fig, ax = plt.subplots(*shape)
    fig.suptitle(title)
    for i,_ in enumerate(features):
        sns.kdeplot(df[features].iloc[:,i],ax=ax[i//4][i%4])
    plt.tight_layout()

def plot_pcs(pcs, colums, shape, title, eps = 0.2):
    fig, ax = plt.subplots(*shape)
    fig.suptitle(title)
    for i in range(pcs.shape[1]):
        filtered = np.array(sorted([list(x) for x in zip(pcs[:,i], colums) if abs(x[0])> eps], key=lambda x:x[0]))
        x = filtered[:,0].astype(float)
        y = filtered[:,1]
        if len(ax.shape)>1: sns.barplot(x=x, y=y,ax=ax[i//shape[1]][i%shape[1]])
        else: sns.barplot(x=x, y=y,ax=ax[i])
    fig.tight_layout()

def one_hot_encode(df: pd.DataFrame, subset_names):
    new_df = df.copy()
    for col_name in subset_names:
        cat_of_tuple = df[col_name].apply(lambda x:  tuple(x.split(".")))
        new_df.drop(col_name, axis=1, inplace=True)
        mlab = MultiLabelBinarizer()
        one_hot_encoded = mlab.fit_transform(cat_of_tuple)
        new_names = [f"{col_name}_{cla}"  for cla in  mlab.classes_]
        new_format = pd.DataFrame(one_hot_encoded, columns=new_names)
        new_df = new_df.reset_index(drop=True).merge(new_format.reset_index(drop=True), left_index=True, right_index=True)
    return new_df