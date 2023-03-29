import pandas as pd

dt = pd.read_csv('steamReviews.csv')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)

print(dt.head())