from huggingface import evaluate as evaluater
import pandas as pd
from sklearn.model_selection import StratifiedKFold

en = True
addition = ""
if en:
    addition = "_en"


path = "data/train" + addition + "_500.csv"
# used between splits to save data to be loaded by evaluation function
train_path = "data/train" + addition + "_temp.csv"
test_path = "data/test" + addition + "_temp.csv"
data = pd.read_csv(path)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
split = skf.split(data, data.label)

metrics = []
for train, test in split:

    data.iloc[train,:].to_csv(train_path, index=False)
    data.iloc[test,:].to_csv(test_path, index=False)
    metrics += [evaluater(train_path, test_path)]

from scipy import stats
target_metric = "eval_accuracy"
target_metric_list = [d[target_metric] for d in metrics]
import numpy as np
target_metric_array = np.asarray(target_metric_list)
mean, sigma = np.mean(target_metric_array), np.std(target_metric_array)
conf_int_a = stats.norm.interval(0.95, loc=mean, scale=sigma)
conf_int_a
