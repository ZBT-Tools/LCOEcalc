import pandas as pd

#df = pd.read_pickle("input4.pkl")
#df.to_excel("test2.xlsx")
#df = pd.read_excel("Dash_LCOE_Configuration.xlsx")
#df_NGfuel_presets = pd.read_excel("Dash_LCOE_NG.xlsx")
#df.loc[:,100]=13
#df.to_pickle("data_return.pkl")
df_input = pd.read_excel("input/Dash_LCOE_Configuration_v4.xlsx",
                         sheet_name=["Systems","Financial","Fuel_NH3","Fuel_NG"])
