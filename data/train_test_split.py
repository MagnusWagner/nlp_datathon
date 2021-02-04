path = "train.csv"
import numpy as np
np.random.seed(123)
addition = ""
en = False
if en:
    addition = "_en"

from sklearn.model_selection import train_test_split
import pandas as pd

path = "train" + addition + ".csv"

data = pd.read_csv(path)
train = train_test_split(data, test_size = 100)
d100 = train[1]
d100.columns = ["", "Narrative", "label"]
d100.to_csv("train" + addition + "_100.csv", index=False)
d500 = train[0]
d500.columns = ["", "Narrative", "label"]
d500.to_csv("train" + addition + "_500.csv", index=False)