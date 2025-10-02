import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('train_data_corrected.csv')
train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train_data_corrected.csv')
dev_df.to_csv('dev_data_corrected.csv')