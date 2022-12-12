""" LCOE Calculation Tool

Description ....


# To Do
    - Show Graphs at website startup. therefore initialize storage with default system data.

Code Structure:

    - Imports
    - Initialization prior to app start
    - App styling and input functions for recurring use in layout
    - App layout definition

"""
import pandas as pd
import dash
from dash import Input, Output, dcc, html, ctx, State, ALL
import dash_bootstrap_components as dbc
import base64

# from docutils.nodes import header
from flask_caching import Cache
import pickle
import jsonpickle
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from scripts.gui_functions import read_input_fields, style_studySettingsCard
from scripts.data_handler import store_data

from scripts.simple_app_gui_functions import style_inpCard_simpleapp
from scripts.simple_app_data_handler import DataHandlerSimpleApp
from scripts.multiplication import DataclassMultiplicationInput, multiplication

# 1. Tool specific definitions & Initialization prior start
# ----------------------------------------------------------------------------------------------------------------------
# Load images (issue with standard image load, due to png?!)
# Fix: https://community.plotly.com/t/png-image-not-showing/15713/2
zbt_png = 'img/logo-zbt-duisburg.png'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

# App initialization
# As GUI for study is build inside callback, checklist id is not available initially, therefore...
# suppress_callback_exceptions=True
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Force Plotly to clear local cache at each start
# Resolves development issue: cached data used instead of updated code
# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.clear()

# 2. App layout
# ----------------------------------------------------------------------------------------------------------------------
# Info: as proposed by dash bootstrap component guide, everything is ordered in dbc.Row's, containing dbc.Col's
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
# "The layout of your app should be built as a series of rows of columns.
#  The Col component should always be used as an immediate child of Row and is a wrapper for your content
#  that ensures it takes up the correct amount of horizontal space."
#  https://getbootstrap.com/docs/5.0/utilities/spacing/


app.layout = dbc.Container([
    # Storage definition
    # ---------------
    # Inputs and results are of small file size, therefore users local memory is used.
    # Limit: 'It's generally safe to store [...] 5~10MB in most desktop-only applications.'
    # https://dash.plotly.com/sharing-data-between-callbacks
    # https://dash.plotly.com/dash-core-components/store
    # dcc.Store(id='storage', storage_type='memory'),

    # Header Row with title & logos
    dbc.Row([dbc.Col(html.H1("This is a simple Application."), width=12, xl=3),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250), width=12, xl=3)]),
    html.Hr(),

    # Main
    dbc.Row([
        # Setting Column
        dbc.Col([
            dbc.Row(dbc.Col(
                style_inpCard_simpleapp(header="Input Parameter",
                                        specific_row_input=[
                                            {"label": "Parameter A", "ident": "a"},
                                            {"label": "Parameter B", "ident": "b"}]
                                        ), width=12)),
            dbc.Row([dbc.Col(dbc.Button(children="run", id="bt_run"), width=2),
                     dbc.Col(dbc.Button("Run Study", id="bt_runstudy"), width=2)]),
            html.Div(id="initStudyCard"),
            html.Div(id="temp"),
            dbc.Row(dbc.Col(dbc.Card(id="studyCard")))
        ], width=12, xl=4),

        # Visualization Column
        dbc.Col([
            dbc.Row(dbc.Col(
                html.Div(id="txt_result"))),
            dbc.Row(dbc.Col(
                dcc.Graph(id='simple_plot')))
        ], width=12, xl=8)
    ])
], fluid=True)


# Callback Functions, app specific
# --------------------------------------------------------------
# --------------------------------------------------------------


@app.callback(
    Output("txt_result", "children"),
    Input("bt_run", "n_clicks"),
    State({"type": "input", 'id': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_nominal_run(*args):
    """
    Input:
        Button Click

    Description:
        1. Collect (nominal) input variables from data fields
        2. Initialize DataHandler, prepare input-sets, perform calculations
        3. Show results


    """
    # 1. Collect nominal input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    datahandler = DataHandlerSimpleApp(df, DataclassMultiplicationInput, parID="id")
    datahandler.create_input_sets()
    datahandler.submit_job()

    # 3. Read results and write into table (could be reworked)
    # ------------------------------------------------------------------------------------------------------------------
    result = datahandler.df_results["result"].item()

    return f"Result of multiplication: {result}"


@app.callback(
    Output("studyCard", "children"),
    Input("initStudyCard", "children"),
    State({"type": "input", 'id': ALL}, 'value'))
def cbf_study_init(*inp):
    df = read_input_fields(ctx.states_list[0])
    pars = [{"label": l, "value": l} for l in df.id]
    card = style_studySettingsCard(header="Sensitivity Study", pars=pars)
    return card


@app.callback(
    Output("simple_plot", "figure"),
    Input("bt_runstudy", "n_clicks"),
    State({"type": "studyInput", "par": "checklist"}, "value"),
    State({"type": "studyInput", "par": "globalVar_perc"}, "value"),
    State({"type": "input", 'id': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_study_run(inp, checklist, perc, data):
    # List of selected variables
    variation_pars = {"checklist": checklist, "variation_perc": perc}

    # 1. Collect nominal input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[2])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    datahandler = DataHandlerSimpleApp(df, DataclassMultiplicationInput, parID="id")
    datahandler.create_input_sets(mode="study", studypar=variation_pars)
    datahandler.submit_job()

    # 3. Read results and simple plot
    # ------------------------------------------------------------------------------------------------------------------
    result = datahandler.df_results
    fig = px.scatter(result, x="a", y="b", hover_data=["result"])

    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
