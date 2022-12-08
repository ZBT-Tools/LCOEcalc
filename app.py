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
from flask_caching import Cache
import pickle
import jsonpickle
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scripts.lcoe_simple import multisystem_calculation
from scripts.data_handler import store_data

from scripts.gui_functions import styling_input_card_component, styling_generic_dropdown, styling_input_card_generic, \
    fill_input_fields, read_input_fields, build_initial_collect

# 1. Tool specific definitions & Initialization prior start
# ----------------------------------------------------------------------------------------------------------------------

# Note: Storage elements (dcc.Store) are defined inside app layout below

system_components = ["HiPowAR", "ICE", "SOFC"]

# Input definition table with presets, excel table
df_input = pd.read_excel("input/Dash_LCOE_ConfigurationV3.xlsx",
                         sheet_name=["Systems", "Financial", "Fuel_NH3", "Fuel_NG"])

# Load images (issue with standard image load, due to png?!)
# Fix: https://community.plotly.com/t/png-image-not-showing/15713/2
hipowar_png = 'img/Logo_HiPowAR.png'
hipowar_base64 = base64.b64encode(open(hipowar_png, 'rb').read()).decode('ascii')
eu_png = 'img/EU_Logo.png'
eu_base64 = base64.b64encode(open(eu_png, 'rb').read()).decode('ascii')
zbt_png = 'img/logo-zbt-duisburg.png'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

# App initialization
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Force Plotly to clear local cache at each start
# Resolves development issue: cached data used instead of updated code
# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.clear()

# 2. App layout
# ----------------------------------------------------------------------------------------------------------------------
# Info: as proposed by dash bootstrap component guide, everything is ordered in dbc.Row's, containing dbc.Col's

app.layout = dbc.Container([
    # Storage definition
    # ---------------
    # Inputs and results are of small file size, therefore users local memory is used.
    # Limit: 'It's generally safe to store [...] 5~10MB in most desktop-only applications.'
    # https://dash.plotly.com/sharing-data-between-callbacks
    # https://dash.plotly.com/dash-core-components/store
    dcc.Store(id='storage', storage_type='memory'),

    # Header Row with title & logos
    dbc.Row([dbc.Col(html.H1("HiPowAR LCOE Tool"), width=4),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=300)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250))]),
    html.Hr(),

    # Accordeon-like User Interface and result presentation
    dbc.Row([dbc.Col([
        dbc.Accordion([
            dbc.AccordionItem(title="Preset Selection", children=[
                # Menu with different drop down menus for preset selections, 'Calculate' Button
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
            dbc.AccordionItem(title="Energy Conversion System Settings", children=[
                # Menu with input cards for each energy conversion system (HiPowAR, SOFC,ICE)
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
            dbc.AccordionItem(title="Environmental Settings", children=[
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
                                                           {'par': 'fuel_cost_Eur_per_kWh',
                                                            'title': "NH3 cost [€/kWh]"},
                                                           {'par': 'fuel_costIncrease_percent_per_year',
                                                            'title': "NH3 cost increase [%/yr]"}]),
                        )),

                        dbc.Row(dbc.Col(
                            styling_input_card_generic(component='Fuel_NG', header="NG Fuel Cost",
                                                       rowinputs=[
                                                           {'par': 'fuel_cost_Eur_per_kWh', 'title': "NG cost [€/kWh]"},
                                                           {'par': 'fuel_costIncrease_percent_per_year',
                                                            'title': "NG cost increase [%/yr]"}])
                        ))

                    ], md=4)
                ])
            ]),
            dbc.AccordionItem(title='Nominal Results', children=[
                dbc.Row(dbc.Col(
                    dbc.Table(id="table_lcoe_nominal", bordered=True)
                ))
            ],),
            dbc.AccordionItem(title="LCOE Study Results", children=[
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
            ], ),
            dbc.AccordionItem(title="About", children=[]),
            dbc.AccordionItem(title="Developer", children=[
                dbc.Row([dbc.Col(dbc.Button("Build: Initial Data Collect", id="bt_collect"), width=2),
                         dbc.Col(dbc.Button("Build: Random Fill Fields", id="bt_fill"), width=2),
                         dbc.Col(dbc.Button("Work: Update Data Collect", id="bt_update_collect"), width=2),
                         dbc.Col(dbc.Button("Debug: Init", id="bt_init"), width=2)
                         ]),
                dbc.Row([html.Pre("Nominal Calculation Done:", id="flag_nominal_calculation_done")]),
                dbc.Row([html.Pre("Sensitivity Calculation Done:", id="flag_sensitivity_calculation_done")]),
                dbc.Row([html.Pre("Build: Initial Collect Input", id="txt_build1")]),
                dbc.Row([html.Pre("Build: Update Collect Input", id="txt_build2")]),
                dbc.Row([html.Pre("Debug Calculation:", id="txt_dev_button_init")])
            ]),
        ], always_open=True)
    ])])
], fluid=True)

# Callback Functions, app specific
# --------------------------------------------------------------
# --------------------------------------------------------------

@app.callback(
    Output("txt_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'HiPowAR', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'SOFC', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'ICE', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(df_input["Systems"].columns[4:]))], )
def cbf_quickstart_select_system_preset(*inp):
    """
    Description:
        Dropdown menu with system presets consist of elements with ids dd_preset_{n}
    Input:
        Each dropdown element triggers callback.
    Output:
    - Output[0]:  Selection name --> text field next to dropdown
    - Output[1:]: System data --> data flieds
    """
    try:
        selection_title = df_input["Systems"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Systems"].columns[4:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Systems"], output=ctx.outputs_list[1:])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_Financial_Selection", "children"),
    Output({'type': 'input', 'component': 'Financials', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_Financial_{n}", "n_clicks") for n in range(len(df_input["Financial"].columns[4:]))], )
def cbf_quickstart_select_financial(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Financial"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
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
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NH3"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
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
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NG"].columns[4:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
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
    data = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "nominal")

    # Read results and write into Table
    # ------------------------------------------------------------------------------------------------------------------
    list_systemname = []
    list_lcoeval = []
    for key, system in data.items():
        list_systemname.append(key)
        list_lcoeval.append(system.df_results.loc["nominal", "LCOE"])

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

    # 2. Initialize systems, prepare input-sets, perform calculations, store data
    # ------------------------------------------------------------------------------------------------------------------
    systems = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "all_minmax")

    # 5. Store data in dcc.storage object
    # -----------------------------------------------------------------------------------------------------------------
    # Create json file:
    data = store_data(systems)
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
    y0 = systems["HiPowAR_NH3"].df_results["LCOE"]
    y1 = systems["SOFC_NH3"].df_results["LCOE"]
    y2 = systems["ICE_NH3"].df_results["LCOE"]
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
    y0 = systems["HiPowAR_NG"].df_results["LCOE"]
    y1 = systems["SOFC_NG"].df_results["LCOE"]
    y2 = systems["ICE_NG"].df_results["LCOE"]
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

        tb = systems[system].df_results.copy()
        tb = tb.apply(pd.to_numeric, errors='ignore')  # Todo relocate

        # Create first plot with only system parameters, identified by "p".

        variation_pars = tb.columns.drop(["size_kW", "LCOE", "name", "fuel_name"])

        # variation_pars = variation_pars.drop([x for x in variation_pars if x[0] != "p"])

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
            # In case modpar has no variation (e.g. fuel cost increase is set as [0,0,0], all values are the same.
            # Thus following rows could include "nominal" set again. This needs to be prevented.
            tbred.drop(index="nominal", inplace=True)
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
            fig.add_trace(trace, row=1, col=1)

        fig.add_hline(y=result_df.loc["nominal", "LCOE"], line_color=colordict[system])

        # # Create second plot with only non-system inherent parameters, identified by not "p".
        #
        # tb = systems[system].lcoe_table.copy()
        # # result_df = pd.DataFrame(columns=["modpar"])
        #
        # variation_pars = tb.columns.drop(["p_size_kW", "LCOE"])
        # variation_pars = variation_pars.drop([x for x in variation_pars if x[0] == "p"])
        #
        # result_df = pd.DataFrame(columns=tb.columns)
        # result_df.loc["nominal"] = tb.loc["nominal"]
        #
        # for modpar in variation_pars:
        #     # Create query string:
        #     qs = ""
        #     cond = [f"{parm} == {result_df.loc['nominal', parm]}" for parm in variation_pars.drop(modpar)]
        #     for c in cond:
        #         qs = qs + c + " & "
        #     qs = qs[:-3]
        #     tbred = tb.query(qs)
        #     rw = tbred.nsmallest(1, modpar)
        #     rw["modpar"] = modpar
        #     result_df = pd.concat([result_df, rw])
        #     rw = tbred.nlargest(1, modpar)
        #     rw["modpar"] = modpar
        #     result_df = pd.concat([result_df, rw])
        #
        # result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc["nominal", "LCOE"]
        #
        # for name, group in result_df.groupby('modpar'):
        #     trace = go.Box()
        #     trace.name = system
        #     trace.x = [name] * 3
        #     trace.y = [result_df.loc["nominal", "LCOE"],
        #                group["LCOE"].max(),
        #                group["LCOE"].min()]
        #     trace.marker["color"] = colordict[system]
        #     # trace.error_y = dict(
        #     #    type='data',
        #     #    symmetric=False,
        #     #    array=[group["diff"].max()],
        #     #    arrayminus=[abs(group["diff"].min())])
        #     fig.add_trace(trace, row=1, col=2, )
        #
        # fig.add_hline(y=result_df.loc["nominal", "LCOE"], line_color=colordict[system])

    fig.update_layout(
        showlegend=False,
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    return fig


@app.callback(
    Output("txt_build1", "children"),
    Input("bt_collect", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_dev_button_build_initialCollectInput(*args):
    """
    Creates new DataFrame / excel table with all inputfields of types defined in callback above.
    """
    df = build_initial_collect(ctx.states_list[0])
    df.to_pickle("input4.pkl")
    df.to_excel("input4.xlsx")

    return "ok"


# # INFO: Function is commented, because there would be an output overlap. Decomment when building GUI!
# @app.callback(
#     Output({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
#     Input("bt_fill", "n_clicks"),
#     State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
#     prevent_initial_call=True)
# def cbf_dev_button_build_randomFillFields(*args):
#     """
#     Fill all fields of type 'input' with random numbers
#     """
#     returnvalues = build_randomfill_input_fields(ctx.states_list[0])
#
#     return returnvalues

@app.callback(
    Output("txt_build2", "children"), Input("bt_update_collect", "n_clicks"),
    State({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
    State({'type': 'input_SOFC', 'index': ALL}, 'value'),
    State({'type': 'input_ICE', 'index': ALL}, 'value'),
    State({'type': 'input_Financials', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NH3', 'index': ALL}, 'value'),
    State({'type': 'input_Fuel_NG', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_dev_button_build_updateCollectInput(inp, *args):
    """
    Intention: Save new parameterset to table.

    ToDo: Implement correctly!
    """
    df = pd.read_pickle("input4.pkl")
    for key, val in ctx.states.items():
        df.loc[key, inp] = val
    df.to_pickle("input4_upd.pkl")
    df.to_excel("input4_upd.xlsx")
    return "ok"


@app.callback(
    Output("txt_dev_button_init", "children"), Input("bt_init", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True)
def cbf_dev_button_init(inp, *args):
    """
    Debug parameter study
    """
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])
    data = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "all_minmax")
    print("ok")


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
