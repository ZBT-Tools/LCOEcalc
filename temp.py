import pandas as pd
import jsonpickle
from itertools import product
import plotly.graph_objects as go

# ToDo:
# Table an der richtigen position numerisch machen
#
#
#


ta = pd.read_pickle("table.pkl")
ta = ta.apply(pd.to_numeric)


pars = ta.columns.drop(["p_size_kW", "LCOE"])

df = pd.DataFrame(columns=ta.columns)
df.loc["nominal"] = ta.loc["nominal"]

for modpar in pars:
    # Create query string:
    qs = ""
    cond = [f"{parm} == {df.loc['nominal',parm]}" for parm in pars.drop(modpar)]
    for c in cond:
        qs = qs + c + " & "
    qs = qs[:-3]
    dfred = ta.query(qs)
    rw = dfred.nsmallest(1, modpar)
    rw["modpar"]=modpar
    df = pd.concat([df, rw])
    rw = dfred.nlargest(1, modpar)
    rw["modpar"]=modpar
    df = pd.concat([df, rw])

df.loc[:,"diff"] = df["LCOE"] - df.loc["nominal","LCOE"]

fig2 = go.Figure()
for name, group in df.groupby('modpar'):
    trace = go.Scatter()
    trace.name = name
    trace.x = [name]
    trace.y = [df.loc["nominal", "LCOE"]]
    trace.error_y = dict(
        type='data',
        symmetric=False,
        array=[group["diff"].max()],
        arrayminus=[abs(group["diff"].min())])
    fig2.add_trace(trace)

fig2.write_html('first_figure.html', auto_open=True)
