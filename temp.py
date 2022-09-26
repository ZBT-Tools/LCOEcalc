import pandas as pd

#df = pd.read_pickle("data.pkl")

df = pd.read_excel("Dash_LCOE_Configuration.xlsx")
df_NGfuel_presets = pd.read_excel("Dash_LCOE_NG.xlsx")
#df.loc[:,100]=13
#df.to_pickle("data_return.pkl")