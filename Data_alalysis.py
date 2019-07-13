import pandas as pd

data_dir="trainingData.csv"

data=pd.read_csv(data_dir)

des=data.describe()
print(des)