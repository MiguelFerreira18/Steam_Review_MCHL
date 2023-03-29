import pandas as pd

dt = pd.read_csv('steamReviews.csv', nrows=100)

print(len(dt.dropna()))
# print(dt.columns)
