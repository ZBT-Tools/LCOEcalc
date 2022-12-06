""" LCOE Calculation Tool

Description ....
Test
#ToDo: Show Graphs at website startup. therefore initialize storage with default system data and remove 'prevent
    callback' from plot callbacks

Code Structure:

    - Imports
    - Initialization prior to app start
    - App styling and input functions for recurring use in layout
    - App layout definition

"""

import dash
from dash import Input, Output, dcc, html, ctx, State, MATCH, ALL
import json
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import plotly.graph_objs as go
from flask_caching import Cache
from dacite import from_dict
import pickle
import jsonpickle
import datetime
import numpy as np
from itertools import product
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scripts.lcoe import SystemIntegrated
from scripts.data_transfer import DC_FinancialInput, DC_SystemInput, DC_FuelInput
from scripts.gui_functions import *

# Definition variables
system_components = ["HiPowAR", "ICE", "SOFC"]

# Initialization prior to app start
# ----------------------------------------------------------------------------------------------------------------------
# Storage is defined as first element inside app layout!

# Read input data, presets from excel definition table
df_input = pd.read_excel("input/Dash_LCOE_ConfigurationV3.xlsx",
                         sheet_name=["Systems", "Financial", "Fuel_NH3", "Fuel_NG"])

# Load images (issue with standard image load, due to png?!)
# https://community.plotly.com/t/png-image-not-showing/15713/2
hipowar_png = 'img/Logo_HiPowAR.png'
hipowar_base64 = base64.b64encode(open(hipowar_png, 'rb').read()).decode('ascii')
eu_png = 'img/EU_Logo.png'
eu_base64 = base64.b64encode(open(eu_png, 'rb').read()).decode('ascii')
zbt_png = 'img/logo-zbt-duisburg.png'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Force Plotly to clear local cache at each start
# Issue occured during development: cached data used instead of updated code
# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.clear()

# App layout definition
# ----------------------------------------------------------------------------------------------------------------------
# Info: as proposed by dash bootstrap component guide, everything is ordered in dbc.Row's, containing dbc.Col's

app.layout = dbc.Container([
    # Inputs and results are of small file size, therefore users local memory is used.
    # Limit: 'It's generally safe to store [...] 5~10MB in most desktop-only applications.'
    # https://dash.plotly.com/sharing-data-between-callbacks
    # https://dash.plotly.com/dash-core-components/store
    dcc.Store(id='storage', storage_type='memory'),
    dcc.Store(id='storage_NG', storage_type='memory'),

    # Header Row with Title, Logos,...
    dbc.Row([dbc.Col(html.H1("HiPowAR LCOE Tool"), width=4),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=300)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250))]),
    html.Hr(),

    # Accordeon-like User Interfalce
    dbc.Row([dbc.Col([
        dbc.Accordion([
            # Menu with different drop down menus for preset selections
            dbc.AccordionItem(title="Preset Selection", children=[
                # Dropdown System Preset Selection
                dbc.Row([
                    dbc.Col(styling_generic_dropdown(id_name="dd_preset", label="System Presets",
                                                     elements=df_input["Systems"].columns[4:]), width=2),
                    dbc.Col(html.P(df_input["Systems"].columns[4], id="txt_Preset_Selection"), width=8)]),
                # Dropdown Financial Preset Selection
                dbc.Row([
                    dbc.Col(styling_generic_dropdown(id_name="dd_Financial", label="Financial Presets",
                                                     elements=df_input["Financial"].columns[4:]), width=2),
                    dbc.Col(html.P(df_input["Financial"].columns[4], id="txt_Financial_Selection"), width=8)]),
                # Dropdown NH3 Fuel Cost Preset Selection
                dbc.Row([
                    dbc.Col(styling_generic_dropdown(id_name="dd_NH3_fuel_cost", label="NH3 Cost Selector",
                                                     elements=df_input["Fuel_NH3"].columns[4:]), width=2),
                    dbc.Col(html.P(df_input["Fuel_NH3"].columns[4], id="txt_NH3_fuel_cost_Preset_Selection"))]),
                # Dropdown NG Fuel Cost Preset Selection
                dbc.Row([
                    dbc.Col(styling_generic_dropdown(id_name="dd_NG_fuel_cost", label="NG Cost Selector",
                                                     elements=df_input["Fuel_NG"].columns[4:]), width=2),
                    dbc.Col(html.P(df_input["Fuel_NG"].columns[4], id="txt_NG_fuel_cost_Preset_Selection"))]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dbc.Button("Run Nominal", id="bt_run_nominal"), width=2),
                    dbc.Col(dbc.Button("Run Study", id="bt_run_study"), width=2)
                ])
            ]),
            # Menu with input cards for each energy conversion system
            dbc.AccordionItem(title="Energy Conversion System Definition I", children=[
                dbc.Row([
                    dbc.Col(styling_input_card_component(component="HiPowAR", header="HiPowAR"), md=4),
                    dbc.Col(styling_input_card_component("SOFC", header="SOFC",
                                                         add_rows=[{"par": "stacklifetime_hr",
                                                                    "title": "Stack Lifetime [hr]"},
                                                                   {"par": "stackexchangecost_percCapex",
                                                                    'title': "Stack Exchange Cost [% Capex]"}]), md=4),
                    dbc.Col(styling_input_card_component(component="ICE", header="Internal Combustion Eng."), md=4)
                ], )
            ], ),
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col(styling_input_card_generic(component="Financials", header="Financials",
                                                       rowinputs=[
                                                           {'par': "discountrate_perc", 'title': "Discount Rate [%]"},
                                                           {'par': "lifetime_yr", 'title': "Lifetime [y]"},
                                                           {'par': "operatinghoursyearly",
                                                            'title': "Operating hours [hr/yr]"}]
                                                       ), md=4),
                    dbc.Col([
                        dbc.Row(dbc.Col(
                            styling_input_card_generic(component='Fuel_NH3', header="NH3 Fuel Cost",
                                                       rowinputs=[
                                                           {'par': 'cost_Eur_per_kWh', 'title': "NH3 cost [€/kWh]"},
                                                           {'par': 'costIncrease_percent_per_year', 'title': "NH3 cost "
                                                                                                             "increase ["
                                                                                                             "%/yr]"}]),
                        )),

                        dbc.Row(dbc.Col(
                            styling_input_card_generic(component='Fuel_NG', header="NG Fuel Cost",
                                                       rowinputs=[
                                                           {'par': 'cost_Eur_per_kWh', 'title': "NG cost [€/kWh]"},
                                                           {'par': 'costIncrease_percent_per_year',
                                                            'title': "NG cost increase [%/yr]"}])
                        ))

                    ], md=4)
                ])
            ], title="General Settings", ),
            dbc.AccordionItem([
                dbc.Row(dbc.Col(
                    dbc.Table(id="table_lcoe_nominal", bordered=True)
                ))
            ], title='Nominal Results'),
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='graph_lcoe_multi_NH3')
                    ]),
                    dbc.Col([
                        dcc.Graph(id='graph_lcoe_multi_NG')
                    ])]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='lcoe-graph-sensitivity')])
                ])
            ], title="LCOE Study Results"),
            dbc.AccordionItem([], title="About"),
            dbc.AccordionItem([
                dbc.Row([dbc.Col(dbc.Button("Fill Randomly", id="bt_randomfill"), width=2),
                         dbc.Col(dbc.Button("Initial Data Collect", id="bt_collect"), width=2),
                         dbc.Col(dbc.Button("Update Data Collect", id="bt_update_collect"), width=2),
                         dbc.Col(dbc.Button("Load Input", id="bt_load_Input"), width=2),
                         dbc.Col(dbc.Button("Debug Print", id="bt_debugprint"), width=2)
                         ]),
                dbc.Row([html.Pre("...", id="flag_nominal_calculation_done")]),
                dbc.Row([html.Pre("...", id="flag_sensitivity_calculation_done")]),
                dbc.Row([html.Pre("...", id="txt_out1")]),  # ToDo: Tidy up
                dbc.Row([html.Pre("...", id="txt_out2")]),
                dbc.Row([html.Pre("...", id="txt_out3")]),
                dbc.Row([html.Pre("...", id="txt_out4")]),
                dbc.Row([html.Pre("...", id="txt_out7")])
            ], title="Developer"),
        ], always_open=True)

    ])])

], fluid=True)


# Callbacks
# --------------------------------------------------------------
# --------------------------------------------------------------


def initialize_systems(df: pd.DataFrame):
    """
    Create dictionary "components_dict" structure from DataFrame df. Reason: dict can be mapped to dataclass easily.
    Structure of dictionary:
    components_dict = { component1: { par1: [val,val,...], par2: [val,val,...],...}, component2:{...},...}
    components_dict = { "HiPowAR": {"Capex": [70,80,90], "Opex": [10,20,30]}, "SOFC": {...}}
    """

    components_dict = {}
    for c in df.component.unique():
        component_dict = {'name': c}
        # Save all parameters for each component in dictionary
        for p in df.loc[df.component == c, 'par'].unique():
            values = df.loc[(df.component == c) & (df.par == p), 'value'].to_list()
            values = [x for x in values if x is not None]  # Remove "None" entries
            component_dict.update([(p, values)])
        components_dict.update([(c, component_dict)])
    # 2. For each entry in components_dict ( = each row of df) create appropriate Dataclass (DC_SystemInput,
    #    DC_FinancialInput or DC_FuelInput) and add it to dicts dict_dataclass_systems or dict_dataclass_additionals
    # ------------------------------------------------------------------------------------------------------------------
    dict_dataclass_systems = {}
    dict_dataclass_additionals = {}

    for key, dct in components_dict.items():
        if key in system_components:
            dict_dataclass_systems.update({key: from_dict(data_class=DC_SystemInput, data=dct)})
        elif key == "Financials":
            dict_dataclass_additionals.update({key: from_dict(data_class=DC_FinancialInput, data=dct)})
        elif key[:4] == "Fuel":
            dict_dataclass_additionals.update({key: from_dict(data_class=DC_FuelInput, data=dct)})

    # 3. Create System objects with data classes in DC_systems
    # ------------------------------------------------------------------------------------------------------------------
    dict_systems = {}

    for key, dct in dict_dataclass_systems.items():
        # NH3
        dict_systems.update({f"{key}_NH3": SystemIntegrated(dct)})
        dict_systems[f"{key}_NH3"].load_fuel_par(dict_dataclass_additionals["Fuel_NH3"])
        dict_systems[f"{key}_NH3"].load_financial_par(dict_dataclass_additionals["Financials"])
        # NG
        dict_systems.update({f"{key}_NG": SystemIntegrated(dct)})
        dict_systems[f"{key}_NG"].load_fuel_par(dict_dataclass_additionals["Fuel_NG"])
        dict_systems[f"{key}_NG"].load_financial_par(dict_dataclass_additionals["Financials"])

    return dict_systems


def prepare_input_table(dict_systems: dict, mode: str):
    """
    Loop through systems and create lcoe input table
    """
    for key, system in dict_systems.items():
        system.prepare_input_table(mode=mode)
    return None


def run_calculation(dict_systems: dict):
    """
    Loop through systems and run lcoe calculation
    """
    # ------------------------------------------------------------------------------------------------------------------
    for key, system in dict_systems.items():
        system.lcoe_table["LCOE"] = system.lcoe_table.apply(lambda row: system.lcoe(row), axis=1)
        system.lcoe_table = system.lcoe_table.apply(pd.to_numeric)

    return None


def store_data(dict_systems: dict):
    """
    # https://github.com/jsonpickle/jsonpickle, as json.dumps can only handle simple variables, no objects, DataFrames..
    # Info: Eigentlich sollte jsonpickle reichen, um dict mit Klassenobjekten, in denen DataFrames sind, zu speichern,
    #       Es gibt jedoch Fehlermeldungen. Daher wird Datenstruktur vorher in pickle (Binärformat)
    #       gespeichert und dieser anschließend in json konvertiert.
    #       (Konvertierung in json ist notwendig für lokalen dcc storage)
    """
    data = pickle.dumps(dict_systems)
    data = jsonpickle.dumps(data)

    return data


@app.callback(
    Output("txt_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'HiPowAR', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'SOFC', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'ICE', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(df_input["Systems"].columns[4:]))], )
def cbf_quickstart_select_preset(*inp):
    """
    Each element of dropdown list  "dd_...." triggers callback.
    Output:
    - Output[0]:   Text next to dropdown menu
    - Output[1:]: Data as defined in definition table
    """
    try:
        selection_title = df_input["Systems"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:  # Initialization
        selection_title = df_input["Systems"].columns[4:][0]

    return_lists = fill_input_fields(selection_title, df=df_input["Systems"], output=ctx.outputs_list[1:])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_Financial_Selection", "children"),
    Output({'type': 'input', 'component': 'Financials', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_Financial_{n}", "n_clicks") for n in range(len(df_input["Financial"].columns[4:]))], )
def cbf_quickstart_select_financial(*inputs):
    try:
        selection_title = df_input["Financial"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:  # Initialization
        selection_title = df_input["Financial"].columns[4:][0]

    return_lists = fill_input_fields(selection_title, df=df_input["Financial"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NH3_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NH3', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NH3"].columns[4:]))])
def cbf_quickstart_select_NH3fuel_preset(*inputs):
    try:
        selection_title = df_input["Fuel_NH3"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:  # Initialization
        selection_title = df_input["Fuel_NH3"].columns[4:][0]
    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NH3"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NG_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NG', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NG_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NG"].columns[4:]))])
def cbf_quickstart_select_NGfuel_preset(*inputs):
    try:
        selection_title = df_input["Fuel_NG"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:  # Initialization
        selection_title = df_input["Fuel_NG"].columns[4:][0]

    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NG"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("flag_nominal_calculation_done", "children"),
    Output("table_lcoe_nominal", "children"),
    Input("bt_run_nominal", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': 'nominal'}, 'value'),
    prevent_initial_call=True)
def cbf_quickstart_button_runNominalLCOE(*args):
    """
    Returns:    - Datetime to 'flag_sensitivity_calculation_done' textfield.
                - system-objects, results included, to storage(s)
    """
    # 1. Collect nominal input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    dict_systems = initialize_systems(df)
    prepare_input_table(dict_systems, 'nominal')
    run_calculation(dict_systems)

    # Read results
    list_systemname = []
    list_lcoeval = []
    for key, system in dict_systems.items():
        list_systemname.append(key)
        list_lcoeval.append(system.lcoe_table.loc["nominal", "LCOE"])

    # Format table
    table_header = [html.Thead(html.Tr([html.Th("System Name"), html.Th("LCOE [€/kWh")]))]

    rows = [html.Tr([html.Td(n), html.Td(v)]) for n, v in zip(list_systemname, list_lcoeval)]

    table_body = [html.Tbody(rows)]
    table = table_header + table_body  # ToDo Why the heading is not appearing?

    return table


@app.callback(
    Output("flag_sensitivity_calculation_done", "children"),
    Output("storage", "data"),
    Input("bt_run_study", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_quickstart_button_runSensitivityLCOE(*args):
    """
    Returns:    - Datetime to 'flag_sensitivity_calculation_done' textfield.
                - system-objects, results included, to storage(s)
    """
    # 1. Collect all input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    dict_systems = initialize_systems(df)
    prepare_input_table(dict_systems, 'all_minmax')
    run_calculation(dict_systems)

    # 5. Store data in dcc.storage object
    # -----------------------------------------------------------------------------------------------------------------
    # Create json file:
    data = store_data(dict_systems)
    return [datetime.datetime.now(), data]


@app.callback(
    Output('graph_lcoe_multi_NH3', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_NH3_update(inp, state):
    # Read results from storage
    systems = jsonpickle.loads(state)
    systems = pickle.loads(systems)

    # Simple LCOE Comparison Plot
    y0 = systems["HiPowAR_NH3"].lcoe_table["LCOE"]
    y1 = systems["SOFC_NH3"].lcoe_table["LCOE"]
    y2 = systems["ICE_NH3"].lcoe_table["LCOE"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name='HiPowAR',
                         boxpoints='all',
                         marker=dict(color='rgb(160,7,97)'),
                         line=dict(color='rgb(31,148,175)'),

                         ))
    fig.add_trace(go.Box(y=y1, name='SOFC',
                         marker=dict(color='lightseagreen'), boxpoints='all'))
    fig.add_trace(go.Box(y=y2, name='ICE',
                         marker=dict(color='lightskyblue'), boxpoints='all'))

    fig.update_layout(
        title="Levelized Cost of Electricity - Green Ammonia",
        # xaxis_title="",
        yaxis_title="LCOE [€/kW]")
    return fig


@app.callback(
    Output('graph_lcoe_multi_NG', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_NG_update(inp, state):
    # Read from storage
    systems = jsonpickle.loads(state)
    systems = pickle.loads(systems)

    # Simple LCOE Comparison Plot
    y0 = systems["HiPowAR_NG"].lcoe_table["LCOE"]
    y1 = systems["SOFC_NG"].lcoe_table["LCOE"]
    y2 = systems["ICE_NG"].lcoe_table["LCOE"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name='HiPowAR',
                         # marker_color='indianred',
                         boxpoints='all',
                         marker=dict(color='rgb(160,7,97)'),
                         line=dict(color='rgb(31,148,175)')
                         ))
    fig.add_trace(go.Box(y=y1, name='SOFC',
                         marker=dict(color='lightseagreen'), boxpoints='all'))
    fig.add_trace(go.Box(y=y2, name='ICE',
                         marker=dict(color='lightskyblue'), boxpoints='all'))

    fig.update_layout(
        title="Levelized Cost of Electricity - Natural Gas",
        # xaxis_title="",
        yaxis_title="LCOE [€/kW]")
    return fig


@app.callback(
    Output('lcoe-graph-sensitivity', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_Sensitivity_update(inp, state):
    """
    Analysis of influence of single parameter
    ------------------------------------
    (Search in table)
    For each parameter keep all other parameter at nominal value and modify
    single parameter
    Add "modpar" column to lcoe Table, so that groupby can be used for plots
    """
    # Read NH3 data from storage
    systems = jsonpickle.loads(state)
    systems = pickle.loads(systems)

    colordict = {"HiPowAR_NH3": 'rgb(160,7,97)', "SOFC_NH3": 'lightseagreen', "ICE_NH3": 'lightskyblue'}

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        # x_title='Your master x-title',
                        y_title='LOEC [€/kW]',
                        subplot_titles=('System Sensitivity', 'Environment Sensitivity'))

    for system in ["HiPowAR_NH3", "SOFC_NH3", "ICE_NH3"]:

        tb = systems[system].lcoe_table.copy()

        # Create first plot with only system parameters, identified by "p".

        variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
        variation_pars = variation_pars.drop([x for x in variation_pars if x[0] != "p"])

        # Build new dataframe for plotting
        result_df = pd.DataFrame(columns=tb.columns)
        result_df.loc["nominal"] = tb.loc["nominal"]  # Always include nominal calculation
        result_df_temp = result_df.copy()

        for modpar in variation_pars:
            # Create query string:
            qs = ""
            # Next rows create query:
            # Find all result rows, where all other values beside modpar are nomial
            cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
            for c in cond:
                qs = qs + c + " & "
            qs = qs[:-3]  # remove last  " & "

            tbred = tb.query(qs)  # search for rows fullfilling query
            rw = tbred.nsmallest(1, modpar)  # find smallest value of modpar for all results and add to result_df
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])
            rw = tbred.nlargest(1, modpar)  # find largest value of modpar for all results and add to result_df
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])

        result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]  # Calculate difference to
        # nominal

        result_df.drop_duplicates(keep='first', inplace=True)

        for name, group in result_df.groupby('modpar'):
            trace = go.Box()
            trace.name = system
            trace.x = [name] * 3
            trace.y = [result_df.loc["nominal", "LCOE"],
                       group["LCOE"].max(),
                       group["LCOE"].min()
                       ]
            trace.marker["color"] = colordict[system]
            # trace.error_y = dict(
            #    type='data',
            #    symmetric=False,
            #    array=[group["diff"].max()],
            #    arrayminus=[abs(group["diff"].min())])
            fig.add_trace(trace, row=1, col=1)

        fig.add_hline(y=result_df.loc["nominal", "LCOE"], line_color=colordict[system])

        # Create second plot with only non-system inherent parameters, identified by not "p".

        tb = systems[system].lcoe_table.copy()
        # result_df = pd.DataFrame(columns=["modpar"])

        variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
        variation_pars = variation_pars.drop([x for x in variation_pars if x[0] == "p"])

        result_df = pd.DataFrame(columns=tb.columns)
        result_df.loc["nominal"] = tb.loc["nominal"]

        for modpar in variation_pars:
            # Create query string:
            qs = ""
            cond = [f"{parm} == {result_df.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
            for c in cond:
                qs = qs + c + " & "
            qs = qs[:-3]
            tbred = tb.query(qs)
            rw = tbred.nsmallest(1, modpar)
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])
            rw = tbred.nlargest(1, modpar)
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])

        result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]

        for name, group in result_df.groupby('modpar'):
            trace = go.Box()
            trace.name = system
            trace.x = [name] * 3
            trace.y = [result_df.loc["nominal", "LCOE"],
                       group["LCOE"].max(),
                       group["LCOE"].min()]
            trace.marker["color"] = colordict[system]
            # trace.error_y = dict(
            #    type='data',
            #    symmetric=False,
            #    array=[group["diff"].max()],
            #    arrayminus=[abs(group["diff"].min())])
            fig.add_trace(trace, row=1, col=2, )

        fig.add_hline(y=result_df.loc["nominal", "LCOE"], line_color=colordict[system])

    fig.update_layout(
        showlegend=False,
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    return fig


@app.callback(
    Output("txt_out1", "children"),
    Input("bt_collect", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_dev_button_initialCollectInput(*args):
    """
    Creates new dataframe / excel table with all inputfields of types defined in callback above.
    Create DataFrame with all input fields and fill with available input
    """
    df = pd.DataFrame(columns=["component", "par", "parInfo"])
    for dct in ctx.states_list[0]:
        data = {'component': dct["id"]["component"], 'par': dct["id"]["par"], 'parInfo': dct["id"]["parInfo"]}
        try:
            data.update({0: dct["value"]})
        except KeyError:
            data.update({0: None})
        new_row = pd.Series(data)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df.to_pickle("input4.pkl")
    df.to_excel("input4.xlsx")

    return "ok"


@app.callback(
    Output("txt_out2", "children"), Input("bt_update_collect", "n_clicks"),
    State({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
    State({'type': 'input_SOFC', 'index': ALL}, 'value'),
    State({'type': 'input_ICE', 'index': ALL}, 'value'),
    State({'type': 'input_Financials', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NH3', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NG', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_dev_button_updateCollectInput(inp, *args):
    """
    Intention: Save new parameterset to table.

    ToDo: Implement correctly
    """
    df = pd.read_pickle("input4.pkl")
    for key, val in ctx.states.items():
        df.loc[key, inp] = val
    df.to_pickle("input4_upd.pkl")
    df.to_excel("input4_upd.xlsx")
    return "ok"


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
