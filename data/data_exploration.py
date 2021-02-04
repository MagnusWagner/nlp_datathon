import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
data = pd.read_csv("train_100.csv")
data.label.hist(bins =6)
plt.show()