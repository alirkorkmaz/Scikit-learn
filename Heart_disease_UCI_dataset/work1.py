# veri hazırlama süreci
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.datasets import *







# veri setini yükleme ve inceleme adımı

import pandas as pd 

df = pd.read_csv("")
print(df.head())
print(df.info())
print(df.describe())

