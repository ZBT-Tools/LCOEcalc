import pandas as pd

df = pd.read_pickle("data.pkl")

df.loc[:,100]=13
df.to_pickle("data_return.pkl")