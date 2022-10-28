""" LCOE Calculation Tool

Description ....

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
from itertools import product
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scripts.lcoe import System
from scripts.data_transfer import DC_FinancialInput, DC_SystemInput, DC_FuelInput

# Initialization prior to app start
# ----------------------------------------------------------------------------------------------------------------------

# Read input data, presets from excel definition table
df_input = pd.read_excel("input/Dash_LCOE_Configuration.xlsx",
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


# App styling and input functions for recurring use
# ----------------------------------------------------------------------------------------------------------------------

# Parameter input row with single input "nominal"
# def input_row_nom(component, ident, text, disabled=None):
#     print("!!! DEPRECATED: input_row_nom")
#     if disabled is None:
#         disabled = [False]
#     row = dbc.Row([
#         dbc.Col(dbc.Label(text), width=6),
#         dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}"}, type="text", size="sm",
#                           disabled=disabled[0]), width=2), ])
#     return row


# Parameter input row with three inputs "nominal","min","max"
# def input_row_nom_min_max(component, ident, text, disabled=None):
#     print("!!! DEPRECATED: input_row_nom_min_max")
#     if disabled is None:
#         disabled = [False, False, False]
#     row = dbc.Row([
#         dbc.Col(dbc.Label(text), width=6),
#         dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}"}, type="number", size="sm",
#                           disabled=disabled[0]), width=2),
#         dbc.Col(
#             dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}_min"}, type="number", disabled=disabled[1],
#                       size="sm"), width=2),
#         dbc.Col(
#             dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}_max"}, type="number", disabled=disabled[2],
#                       size="sm"), width=2)])
#     return row


# Generalized input row
def input_row_generic(component: str, ident: str, text: str, n_inputfields: int,
                      postfixes: list = None, widths: list = None,
                      disabled: list = None) -> dbc.Row:
    """
    :param component:
    :param ident:
    :param text:
    :param n_inputfields:
    :param postfixes:
    :param widths:
    :param disabled:
    :return:
    """
    # Default column/field widths
    if widths is None:
        if n_inputfields == 1:
            widths = [6, 2]
        elif n_inputfields == 2:
            widths = [6, 2, 2]
        elif n_inputfields == 3:
            widths = [6, 2, 2, 2]
        else:
            equal_wd = int(12 / n_inputfields)
            widths = [equal_wd] * n_inputfields

    # Default postfixes
    if postfixes is None:
        if n_inputfields == 1:
            postfixes = [""]  # Nominal value only, no postfix required
        elif n_inputfields == 3:
            postfixes = ["", "_min", "_max"]
        else:
            postfixes = [f"_{n}" for n in range(n_inputfields)]

    if disabled is None:
        disabled = [False] * n_inputfields

    row_columns = [dbc.Col(dbc.Label(text), width=6)]

    for w, d, p in zip(widths, disabled, postfixes):
        col = dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}{p}"}, type="number", size="sm",
                                disabled=d), width=w),
        row_columns.append(col)

    return dbc.Row(row_columns)


# Generalized system definition card, containing multiple input rows
# def old_card_component_input(name: str, add_items: dict = None) -> dbc.Card:
#     """
#
#     :param name: Card Heading
#     :param add_items: Dict of additional input_row_general() input, structure
#                         {ident: [**kwargs]}
#     :return: dbc.Card
#     """
#     print("Deprecated")
#     card_body_rows = [
#         # Heading Row
#         dbc.Row([
#             dbc.Col(width=6),
#             dbc.Col(dbc.Label("Nominal"), width=2),
#             dbc.Col(dbc.Label("Min"), width=2),
#             dbc.Col(dbc.Label("Max"), width=2)]),
#
#         # Standard system input rows
#         input_row_general(component=name, ident="size_kW", text="El. Output [kW]", n_inputfields=1),
#         input_row_general(component=name, ident="capex_Eur_kW", text="Capex [€/kW]", n_inputfields=3),
#         input_row_general(component=name, ident="opex_Eur_kWh", text="Opex (no Fuel) [€/kWh]", n_inputfields=3),
#         input_row_general(component=name, ident="eta_perc", text="Efficiency [%]", n_inputfields=3)]
#
#     # Add additional rows
#     list_additional_rows = []
#     for key, val in add_items.items():
#         rw = input_row_general(component=name, ident=key, **val)
#         card_body_rows.append(rw)
#         # list_additional_rows.append(rw)
#         # card_body_rows.extend(list_additional_rows)
#     card = dbc.Card([
#         dbc.CardHeader(f"{name}"),
#         dbc.CardBody([
#             html.Div(card_body_rows)])])
#     return card


# General Card definition with input rows
def input_card_generic(component: str, header: str, rowinputs: dict) -> dbc.Card:
    """
    :param rowinputs:
    :param component:
    :param header:
    :return:
    """

    # LCOE Tool specific column definition: 4 Columns
    rows = [dbc.Row([dbc.Col(width=6),
                     dbc.Col(dbc.Label("Nominal"), width=2),
                     dbc.Col(dbc.Label("Min"), width=2),
                     dbc.Col(dbc.Label("Max"), width=2)
                     ])]
    rws = [input_row_generic(component=component, ident=key, text=val["text"], n_inputfields=3) for key, val in
           rowinputs]
    rows.extend(rws)

    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody([
            html.Div([
                dbc.Row(rows)
            ])])])
    return card


def input_card_component(component: str, header: str, rowinputs: dict = None) -> dbc.Card:
    # Component Specific input
    extended_rowInput = {"size_kW": {"text": "El. Output [kW]", "n_inputfields": 1},
                         "capex_Eur_kW": {"text": "Capex [€/kW]", "n_inputfields": 3},
                         "opex_Eur_kWh": {"text": "Opex (no Fuel) [€/kWh]", "n_inputfields": 3},
                         "eta_perc": {"text": "Efficiency [%]", "n_inputfields": 3}}
    extended_rowInput.update(rowinputs)

    card = input_card_generic(component, header, extended_rowInput)
    return card


def generic_dropdown(id_name: str, label: str, elements: list) -> dbc.DropdownMenu:
    """

    :param id_name: dash component name
    :param label: label
    :param elements: list of dropdown menu items, ID is generated like {id_name}_{list counter}
    :return: 
    """
    dropdown = dbc.DropdownMenu(
        id=id_name,
        label=label,
        children=[dbc.DropdownMenuItem(el, id=f"{id_name}_{ct}", n_clicks=0) for ct, el in enumerate(elements)]
    )
    return dropdown


# App layout definition
# ----------------------------------------------------------------------------------------------------------------------
# Info: as proposed by dash bootstrap component guide, everything is ordered in dbc.Row's, containing dbc.Col's

app.layout = dbc.Container([
    # Inputs and results are of small file size, therefore users local memory is used.
    # Limit: It's generally safe to store [...] 5~10MB in most desktop-only applications.
    # https://dash.plotly.com/sharing-data-between-callbacks
    # https://dash.plotly.com/dash-core-components/store
    dcc.Store(id='storage', storage_type='memory'),

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
                    dbc.Col(generic_dropdown(id="dd_preset", label="System Presets",
                                             elements=df_input["Systems"].columns[3:]), width=2),
                    dbc.Col(html.P(df_input["Systems"].columns[3], id="txt_Preset_Selection"), width=8)]),
                # Dropdown Financial Preset Selection
                dbc.Row([
                    dbc.Col(generic_dropdown(id="dd_Financial", label="Financial Presets",
                                             elements=df_input["Financial"].columns[3:]), width=2),
                    dbc.Col(html.P(df_input["Financial"].columns[3], id="txt_Financial_Selection"), width=8)]),
                # Dropdown NH3 Fuel Cost Preset Selection
                dbc.Row([
                    dbc.Col(generic_dropdown(id="dd_NH3_fuel_cost", label="NH3 Cost Selector",
                                             elements=df_input["Fuel_NH3"].columns[3:]), width=2),
                    dbc.Col(html.P(df_input["Fuel_NH3"].columns[3], id="txt_NH3_fuel_cost_Preset_Selection"))]),
                # Dropdown NG Fuel Cost Preset Selection
                dbc.Row([
                    dbc.Col(generic_dropdown(id="dd_NG_fuel_cost", label="NG Cost Selector",
                                             elements=df_input["Fuel_NG"].columns[3:]), width=2),
                    dbc.Col(html.P(df_input["Fuel_NG"].columns[3], id="txt_NG_fuel_cost_Preset_Selection"))])
            ]),

            # Menu with input cards for each energy conversion system
            dbc.AccordionItem(title="Energy Conversion System Definition I", children=[
                dbc.Row([
                    dbc.Col(card_component_input("HiPowAR"), md=4),
                    dbc.Col(card_component_input("SOFC", add_items={"stacklifetime_hr": "Stack Lifetime [hr]",
                                                                    "stackexchangecost_percCapex": "Stack Exchange Cost [% Capex]"}),
                            md=4),
                    dbc.Col(card_component_input("ICE"), md=4),
                ], )
            ], ),
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col(card_generic_input(component="Financials", header="Financials",
                                               ident=["discountrate_perc", "lifetime_yr", "operatinghoursyearly"],
                                               text=["Discount Rate [%]",
                                                     "Lifetime [y]",
                                                     "Operating hours [hr/yr]"], ), md=4),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Fuel Cost Settings"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(width=6),
                                dbc.Col(dbc.Label("Nominal"), width=2),
                                dbc.Col(dbc.Label("Min"), width=2),
                                dbc.Col(dbc.Label("Max"), width=2)]),

                            input_row_nom_min_max(component="Fuel_NH3", ident="cost_Eur_per_kWh",
                                                  text="NH3 cost [€/kWh]"),
                            input_row_nom_min_max(component="Fuel_NH3", ident="costIncrease_percent_per_year",
                                                  text="NH3 cost increase [%/yr]"),
                            html.Hr(),
                            input_row_nom_min_max(component="Fuel_NG", ident="cost_Eur_per_kWh",
                                                  text="NG cost [€/kWh]"),
                            input_row_nom_min_max(component="Fuel_NG", ident="costIncrease_percent_per_year",
                                                  text="NG cost increase [%/yr]")
                        ])
                    ]), md=4
                    ),

                ])
            ], title="General Settings", ),
            dbc.AccordionItem([dbc.Row([
                dbc.Col([
                    dcc.Graph(id='lcoe-graph')
                ])
            ])], title="LCOE Plots"),
            dbc.AccordionItem([
                dbc.Row([
                    # dbc.Col([dcc.Graph(id='lcoe-graph-sensitivity')]),
                    dbc.Col([dcc.Graph(id='lcoe-graph-sensitivity2')]),
                ]),

            ], title="LCOE Sensitivity Study"),
            dbc.AccordionItem([], title="About"),
            dbc.AccordionItem([
                dbc.Row([dbc.Col(dbc.Button("Initial Data Collect", id="bt_collect"), width=2),
                         dbc.Col(dbc.Button("Update Data Collect", id="bt_update_collect"), width=2),
                         dbc.Col(dbc.Button("Load Input", id="bt_load_Input"), width=2),
                         dbc.Col(dbc.Button("Process Input", id="bt_process_Input"), width=2),
                         dbc.Col(dbc.Button("Debug Print", id="bt_debugprint"), width=2)
                         ]),
                dbc.Row([html.Pre("...", id="txt_out1")]),  # ToDo
                dbc.Row([html.Pre("...", id="txt_out2")]),
                dbc.Row([html.Pre("...", id="txt_out3")]),
                dbc.Row([html.Pre("...", id="txt_out4")]),
                dbc.Row([html.Pre("...", id="txt_out5")]),
                dbc.Row([html.Pre("...", id="txt_out6")]),
                dbc.Row([html.Pre("...", id="txt_out7")])
            ], title="Developer"),
        ], always_open=True)

    ])])

], fluid=True)


# Callbacks
# --------------------------------------------------------------
# --------------------------------------------------------------


def fill_inputfields(input, df):
    return_lists = []
    for li in ctx.outputs_list:
        return_list = []
        for el in li:
            comp = el["id"]["type"][6:]
            par = el["id"]["index"]
            try:
                return_list.append(
                    df.loc[(df.component == comp) & (df.parameter == par), input].item())
            except:
                # print(f"No data found for {comp},{par}")
                return_list.append(None)
        return_lists.append(return_list)
    return return_lists


# Initialization Callback


@app.callback(
    Output("txt_Preset_Selection", "children"),
    [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(df_input["Systems"].columns[3:]))],
    prevent_initial_call=True)
def quickstart_select_preset_I(*inputs):
    selection_id = df_input["Systems"].columns[3:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    Output({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
    Output({'type': 'input_SOFC', 'index': ALL}, 'value'),
    Output({'type': 'input_ICE', 'index': ALL}, 'value'),
    Input("txt_Preset_Selection", "children"),
    # prevent_initial_call=True
)
def quickstart_select_preset_II(input):
    return_lists = fill_inputfields(input, df_input["Systems"])
    return return_lists


@app.callback(
    Output("txt_Financial_Selection", "children"),
    [Input(f"dd_Financial_{n}", "n_clicks") for n in range(len(df_input["Financial"].columns[3:]))],
    prevent_initial_call=True)
def quickstart_select_financial_I(*inputs):
    selection_id = df_input["Financial"].columns[3:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    [Output({'type': 'input_Financials', 'index': ALL}, 'value')],
    Input("txt_Financial_Selection", "children"))
def quickstart_select_financial_II(input):
    return_lists = fill_inputfields(input, df_input["Financial"])
    return return_lists


@app.callback(
    Output("txt_NH3_fuel_cost_Preset_Selection", "children"),
    [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NH3"].columns[3:]))],
    prevent_initial_call=True)
def quickstart_select_NH3fuel_preset_I(*input):
    selection_id = df_input["Fuel_NH3"].columns[3:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    [Output({'type': 'input_Fuel_NH3', 'index': ALL}, 'value')],
    Input("txt_NH3_fuel_cost_Preset_Selection", "children"), )
def quickstart_select_NH3fuel_preset_II(input):
    return_lists = fill_inputfields(input, df_input["Fuel_NH3"])
    return return_lists


@app.callback(
    Output("txt_NG_fuel_cost_Preset_Selection", "children"),
    [Input(f"dd_NG_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NG"].columns[3:]))],
    prevent_initial_call=True)
def quickstart_select_NGfuel_preset_I(*input):
    selection_id = df_input["Fuel_NG"].columns[3:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    [Output({'type': 'input_Fuel_NG', 'index': ALL}, 'value')],
    Input("txt_NG_fuel_cost_Preset_Selection", "children"), )
def quickstart_select_NGfuel_preset_II(input):
    return_lists = fill_inputfields(input, df_input["Fuel_NG"])
    return return_lists


@app.callback(
    Output('lcoe-graph', 'figure'), Input("txt_out6", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def lcoeplot_update(input, state):
    # Read from storage
    systems = jsonpickle.loads(state)
    systems = pickle.loads(systems)
    # print("loaded")

    # Simple LCOE Comparison Plot
    y0 = systems["HiPowAR"].lcoe_table["LCOE"]
    y1 = systems["SOFC"].lcoe_table["LCOE"]
    y2 = systems["ICE"].lcoe_table["LCOE"]
    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name='HiPowAR',
                         # marker_color='indianred',
                         boxpoints='all',
                         marker_color='rgb(160,7,97)',
                         line_color='rgb(31,148,175)'
                         ))
    fig.add_trace(go.Box(y=y1, name='SOFC',
                         marker_color='lightseagreen', boxpoints='all'))
    fig.add_trace(go.Box(y=y2, name='ICE',
                         marker_color='lightskyblue', boxpoints='all'))

    fig.update_layout(
        title="Levelized Cost of Electricity ",
        # xaxis_title="",
        yaxis_title="LCOE [€/kW]")
    return fig


# @app.callback(
#     Output('lcoe-graph-sensitivity', 'figure'), Input("txt_out6", "children"),
#     State('storage', 'data'),
#     prevent_initial_call=True)
# def lcoesensitivity_plot_sensitivity_update(input, state):
#     # Read from storage
#     systems = jsonpickle.loads(state)
#     systems = pickle.loads(systems)
#     print("loaded")
#
#     # Influence of single parameter
#     # ------------------------------------
#     # (Search in table)
#     # For each parameter keep all other parameter at nominal value and modify
#     # single parameter
#     # Add "modpar" column to lcoe Table, so that groupby can be used for plots
#     # #ToDO: Here or in lcoe.py?
#
#     fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
#
#     tb = systems["HiPowAR"].lcoe_table.copy()
#     # result_df = pd.DataFrame(columns=["modpar"])
#
#     variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
#     variation_pars = variation_pars.drop([x for x in variation_pars if x[0] != "p"])
#
#     result_df = pd.DataFrame(columns=tb.columns)
#     result_df.loc["nominal"] = tb.loc["nominal"]
#
#     for modpar in variation_pars:
#         # Create query string:
#         qs = ""
#         cond = [f"{parm} == {result_df.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
#         for c in cond:
#             qs = qs + c + " & "
#         qs = qs[:-3]
#         tbred = tb.query(qs)
#         rw = tbred.nsmallest(1, modpar)
#         rw["modpar"] = modpar
#         result_df = pd.concat([result_df, rw])
#         rw = tbred.nlargest(1, modpar)
#         rw["modpar"] = modpar
#         result_df = pd.concat([result_df, rw])
#
#     result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]
#
#     # fig = go.Figure()
#     for name, group in result_df.groupby('modpar'):
#         trace = go.Scatter()
#         trace.name = name
#         trace.x = [name]
#         trace.y = [result_df.loc["nominal", "LCOE"]]
#         trace.error_y = dict(
#             type='data',
#             symmetric=False,
#             array=[group["diff"].max()],
#             arrayminus=[abs(group["diff"].min())])
#         fig.add_trace(trace, row=1, col=1)
#
#     fig.add_hline(y=result_df.loc["nominal", "LCOE"])
#
#     ###################################################################################################################
#
#     tb = systems["HiPowAR"].lcoe_table.copy()
#     # result_df = pd.DataFrame(columns=["modpar"])
#
#     variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
#     variation_pars = variation_pars.drop([x for x in variation_pars if x[0] == "p"])
#
#     result_df = pd.DataFrame(columns=tb.columns)
#     result_df.loc["nominal"] = tb.loc["nominal"]
#
#     for modpar in variation_pars:
#         # Create query string:
#         qs = ""
#         cond = [f"{parm} == {result_df.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
#         for c in cond:
#             qs = qs + c + " & "
#         qs = qs[:-3]
#         tbred = tb.query(qs)
#         rw = tbred.nsmallest(1, modpar)
#         rw["modpar"] = modpar
#         result_df = pd.concat([result_df, rw])
#         rw = tbred.nlargest(1, modpar)
#         rw["modpar"] = modpar
#         result_df = pd.concat([result_df, rw])
#
#     result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]
#
#     # fig = go.Figure()
#     for name, group in result_df.groupby('modpar'):
#         trace = go.Scatter()
#         trace.name = name
#         trace.x = [name]
#         trace.y = [result_df.loc["nominal", "LCOE"]]
#         trace.error_y = dict(
#             type='data',
#             symmetric=False,
#             array=[group["diff"].max()],
#             arrayminus=[abs(group["diff"].min())])
#         fig.add_trace(trace, row=1, col=2, )
#
#     return fig


@app.callback(
    Output('lcoe-graph-sensitivity2', 'figure'), Input("txt_out6", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def lcoesensitivity_plot_sensitivity2_update(input, state):
    # Read from storage
    systems = jsonpickle.loads(state)
    systems = pickle.loads(systems)
    # print("loaded")

    # Influence of single parameter
    # ------------------------------------
    # (Search in table)
    # For each parameter keep all other parameter at nominal value and modify
    # single parameter
    # Add "modpar" column to lcoe Table, so that groupby can be used for plots
    # #ToDO: Here or in lcoe.py?
    colordict = {"HiPowAR": 'rgb(160,7,97)', "SOFC": 'lightseagreen', "ICE": 'lightskyblue'}

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        # x_title='Your master x-title',
                        y_title='LOEC [€/kW]',
                        subplot_titles=('System Sensitivity', 'Environment Sensitivity'))

    for system in ["HiPowAR", "SOFC", "ICE"]:

        tb = systems[system].lcoe_table.copy()
        # result_df = pd.DataFrame(columns=["modpar"])

        variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
        variation_pars = variation_pars.drop([x for x in variation_pars if x[0] != "p"])

        result_df = pd.DataFrame(columns=tb.columns)
        result_df.loc["nominal"] = tb.loc["nominal"]
        result_df_temp = result_df.copy()

        for modpar in variation_pars:
            # Create query string:
            qs = ""
            cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
            for c in cond:
                qs = qs + c + " & "
            qs = qs[:-3]
            # print(f"Query is:{qs}")
            # print(f"Query end")
            tbred = tb.query(qs)
            rw = tbred.nsmallest(1, modpar)
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])
            rw = tbred.nlargest(1, modpar)
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])

        result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]

        result_df.drop_duplicates(keep='first', inplace=True)

        # fig = go.Figure()
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

        ###################################################################################################################

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

        # fig = go.Figure()
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
        # yaxis_title='LCOE [€/kW]',
        showlegend=False,
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    return fig


@app.callback(
    Output("txt_out1", "children"), Input("bt_collect", "n_clicks"),
    State({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
    State({'type': 'input_SOFC', 'index': ALL}, 'value'),
    State({'type': 'input_ICE', 'index': ALL}, 'value'),
    State({'type': 'input_Financials', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NH3', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NG', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def dev_button_initialCollectInput(*args):
    """
    :param args:
    :return: Creates new dataframe / excel table with all inputfields of types defined in callback above.
    """
    df = pd.DataFrame(index=ctx.states.keys(), columns=["component", "parameter"])
    for key, val in ctx.states.items():
        comp = key.split('"')[7][6:]
        par = key.split('"')[3]
        df.loc[key, "parameter"] = par
        df.loc[key, "component"] = comp
        df.loc[key, 0] = val
    df.to_pickle("input4.pkl")
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
def dev_button_updateCollectInput(input, *args):
    df = pd.read_pickle("input4.pkl")
    for key, val in ctx.states.items():
        df.loc[key, input] = val
    df.to_pickle("input4_upd.pkl")
    df.to_excel("input4_upd.xlsx")
    return "ok"


@app.callback(
    [Output("txt_out6", "children"), Output("storage", "data")], Input("bt_process_Input", "n_clicks"),
    State({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
    State({'type': 'input_SOFC', 'index': ALL}, 'value'),
    State({'type': 'input_ICE', 'index': ALL}, 'value'),
    State({'type': 'input_Financials', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NH3', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NG', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def dev_button_procSelection(*args):
    # 1. Collect all input variables from data fields
    # 2. Save data in DataClasses
    # 3. Initialize System objects
    # 4. Perform LCOE Calculation
    # 5. Save Systems in store locally

    # print('StartProc:', datetime.datetime.now())
    # 1. Collect all input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    dict_combined = {}
    for li in ctx.states_list:
        dict_name = li[0]["id"]["type"][6:]  # "HiPowar",....
        dict_data = {"name": dict_name}

        # Identify unique elements
        listelements = []
        for el in li:
            par = el["id"]["index"]
            if (par[-4:] == "_max") or (par[-4:] == "_min"):
                par = par[:-4]
            listelements.append(par)
        unique_listelements = list(set(listelements))

        # Initialize Data Dictionary
        for le in unique_listelements:
            #
            matching = [s for s in listelements if le in s]
            dict_data.update({le: [None] * len(matching)})

        # Fill Data Dictionary
        for el in li:
            par = el["id"]["index"]
            if par[-4:] == "_max":
                par = par[:-4]
                dict_data[par][-1] = el["value"]
            elif par[-4:] == "_min":
                par = par[:-4]
                dict_data[par][-2] = el["value"]
            else:
                dict_data[par][0] = el["value"]

        dict_combined.update({dict_name: dict_data})
    # print("debug")
    # Initialization of DataClasses
    system_name_list = ["HiPowAR", "ICE", "SOFC"]  # ToDO: Global definition
    DC_systems = {}
    DC_additionals = {}
    # print("debug")

    # 2. Save data in DataClasses
    # ------------------------------------------------------------------------------------------------------------------
    for key, dct in dict_combined.items():
        if dct["name"] in system_name_list:
            DC_systems.update({dct["name"]: from_dict(data_class=DC_SystemInput, data=dct)})
        elif dct["name"] == "Financials":
            DC_additionals.update({dct["name"]: from_dict(data_class=DC_FinancialInput, data=dct)})
        elif dct["name"][:4] == "Fuel":
            DC_additionals.update({dct["name"]: from_dict(data_class=DC_FuelInput, data=dct)})

    # 3. Initialize System objects
    # ------------------------------------------------------------------------------------------------------------------
    systems = {}
    for key, dct in DC_systems.items():
        systems.update({key: System(dct)})

    for key, system in systems.items():
        system.load_fuel_par(DC_additionals["Fuel_NH3"])
        system.load_financial_par(DC_additionals["Financials"])

        # 4. Perform LCOE Calculation'
        # --------------------------------------------------------------------------------------------------------------
        # print('Start Prep:', datetime.datetime.now())
        # system.prep_lcoe_input(mode="all")
        system.prep_lcoe_input(mode="all_minmax")
        # print('Start Calc:', datetime.datetime.now())
        system.lcoe_table["LCOE"] = system.lcoe_table.apply(lambda row: systems[key].lcoe(row), axis=1)
        # print('End Calc:', datetime.datetime.now())
        system.lcoe_table = system.lcoe_table.apply(pd.to_numeric)

    # 5. Store data in dcc.storage object
    # -----------------------------------------------------------------------------------------------------------------
    # https://github.com/jsonpickle/jsonpickle, as json.dumps can only handle simple variables
    # Info: Eigentlich sollte jsonpickle reichen, um dict mit Klassenobjekten, in denen DataFrames sind, zu speichern,
    #       Es gibt jedoch Fehlermeldungen auf Speicher-Pointer-Ebene. Daher wird Datenstruktur vorher in pickle (Binärformat)
    #       gespeichert und dieser anschließend in json konvertiert.
    # Konvertierung in json ist notwendig für lokalen dcc storage.
    #
    data = pickle.dumps(systems)
    data = jsonpickle.dumps(data)
    # load = pickle.loads(pk)

    return [datetime.datetime.now(), data]


@app.callback(
    Output("txt_out7", "children"), Input("bt_debugprint", "n_clicks"),
    State({'type': 'input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NH3_input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NG_input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_debugprint(*args):
    for el in ctx.states_list[0]:
        print(el)


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
