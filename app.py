""" LCOE Calculation Tool

Info
Description
    - Debugging/Developent functions and AccordeonItem are commented out.
# Ideas
    - Show Graphs at website startup. therefore initialize storage with default system data.

Code Structure:

    - Imports
    - Initialization prior to app start
    - App styling and input functions for recurring use in layout
    - App layout definition

"""
# import logging
import logging.config
import pandas as pd
import dash
from dash import Input, Output, dcc, html, ctx, State, ALL
import dash_bootstrap_components as dbc
import base64
# from flask_caching import Cache
import pickle
import jsonpickle
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scripts.lcoe_simple import multisystem_calculation
from scripts.data_handler import store_data
from scripts.gui_functions import fill_input_fields, read_input_fields, build_initial_collect, style_generic_dropdown, \
    style_inpCard_LCOE_comp, style_inpCard_LCOE

# Logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# 1. Tool specific definitions & Initialization prior start
# ----------------------------------------------------------------------------------------------------------------------

# Note: Storage elements (dcc.Store) are defined inside app layout below

system_components = ["HiPowAR", "ICE", "SOFC"]

# Input definition table with presets, excel table
df_input = pd.read_excel("input/Dash_LCOE_ConfigurationV3.xlsx",
                         sheet_name=["Systems", "Financial", "Fuel_NH3", "Fuel_NG"])

first_clm = 5

# Load images (issue with standard image load, due to png?!)
# Fix: https://community.plotly.com/t/png-image-not-showing/15713/2
hipowar_png = 'img/Logo_HiPowAR.png'
hipowar_base64 = base64.b64encode(open(hipowar_png, 'rb').read()).decode('ascii')
eu_png = 'img/EU_Logo.png'
eu_base64 = base64.b64encode(open(eu_png, 'rb').read()).decode('ascii')
zbt_png = 'img/logo-zbt-duisburg.png'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

# App initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Force Plotly to clear local cache at each start
# Resolves development issue: cached data used instead of updated code
# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
# cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
# cache.clear()

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
    dcc.Store(id='storage', storage_type='memory'),

    # Header Row with title & logos
    dbc.Row([dbc.Col(html.H1("UNDER DEVELOPMENT!!! HiPowAR LCOE Tool"), width=12, xl=3),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100), width=12, xl=3),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=300), width=12, xl=3),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250), width=12, xl=3)]),
    html.Hr(),

    # Accordeon-like User Interface and result presentation

    # Main
    dbc.Row([
        # Setting Column
        dbc.Col([
            dcc.Markdown('''
                        #### HiPowAR
                        HIGHLY EFFICIENT POWER PRODUCTION BY GREEN AMMONIA TOTAL OXIDATION IN A MEMBRANE REACTOR
                        
                        https://www.hipowar.eu/home
                        
                        '''),
            dbc.Accordion([
                dbc.AccordionItem(title="Preset Selection", children=[
                    # Menu with different drop down menus for preset selections, 'Calculate' Button
                    # Dropdown System Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_preset", label="System",

                                                       elements=df_input["Systems"].columns[first_clm:]),
                                width=6, xl=4),
                        dbc.Col(html.P(df_input["Systems"].columns[-1], id="txt_Preset_Selection"),
                                width=12, xl=8)]),
                    # Dropdown Financial Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_Financial", label="Financial",
                                                       elements=df_input["Financial"].columns[first_clm:]),
                                width=6, xl=4),
                        dbc.Col(html.P(df_input["Financial"].columns[-1], id="txt_Financial_Selection"),
                                width=12, xl=8)]),
                    # Dropdown NH3 Fuel Cost Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_NH3_fuel_cost", label="NH3 Cost",
                                                       elements=df_input["Fuel_NH3"].columns[first_clm:]),
                                width=6, xl=4),
                        dbc.Col(html.P(df_input["Fuel_NH3"].columns[-1], id="txt_NH3_fuel_cost_Preset_Selection"),
                                width=12, xl=8)]),
                    # Dropdown NG Fuel Cost Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_NG_fuel_cost", label="NG",
                                                       elements=df_input["Fuel_NG"].columns[first_clm:]),
                                width=6, xl=4),
                        dbc.Col(html.P(df_input["Fuel_NG"].columns[-1], id="txt_NG_fuel_cost_Preset_Selection"),
                                width=12, xl=8)]),

                    html.Hr(),
                    dbc.Row([
                        dbc.Col(dbc.Button("Run Nominal", id="bt_run_nominal", size="sm"), width=3),
                        dbc.Col(dbc.Button("Run Study", id="bt_run_study", size="sm"), width=3),

                        dbc.Col(dbc.Spinner(html.Div(id="loading-output"), color="success"), width=3, align="center")
                    ])

                ]),
                dbc.AccordionItem(title="Energy Conversion System Settings", children=[
                    # Menu with input cards for each energy conversion system (HiPowAR, SOFC,ICE)
                    dbc.Row([

                        dbc.Col(style_inpCard_LCOE_comp(header="HiPowAR", component="HiPowAR"), width=12),
                        dbc.Col(style_inpCard_LCOE_comp(header="SOFC", component="SOFC",
                                                        # add_rows=[{"par": "stacklifetime_hr",
                                                        #            "label": "Stack Lifetime [hr]"},
                                                        #           {"par": "stackexchangecost_percCapex",
                                                        #            'label': "Stack Exchange Cost [% Capex]"}]
                                                        ),
                                width=12),

                        dbc.Col(style_inpCard_LCOE_comp(component="ICE", header="Internal Combustion Eng."),

                                width=12)
                    ], )
                ], ),
                dbc.AccordionItem(title="Environmental Settings", children=[
                    dbc.Row([

                        dbc.Col(style_inpCard_LCOE(component="Financials", header="Financials",

                                                   specific_row_input=[
                                                       {'par': "discountrate_perc",
                                                        'label': "Discount Rate [%]"},
                                                       {'par': "lifetime_yr", 'label': "Lifetime [y]"},
                                                       {'par': "operatinghoursyearly",
                                                        'label': "Operating hours [hr/yr]"}]
                                                   ), width=12),
                        dbc.Col([
                            dbc.Row(dbc.Col(

                                style_inpCard_LCOE(header="NH3 Fuel Cost", component='Fuel_NH3',
                                                   specific_row_input=[
                                                       {'par': 'fuel_cost_Eur_per_kWh',
                                                        'label': "NH3 cost [€/kWh]"},
                                                       {'par': 'fuel_costIncrease_percent_per_year',
                                                        'label': "NH3 cost increase [%/yr]"}]), width=12,
                            )),

                            dbc.Row(dbc.Col(
                                style_inpCard_LCOE(header="NG Fuel Cost", component='Fuel_NG',
                                                   specific_row_input=[
                                                       {'par': 'fuel_cost_Eur_per_kWh', 'label': "NG cost ["
                                                                                                 "€/kWh]"},
                                                       {'par': 'fuel_costIncrease_percent_per_year',
                                                        'label': "NG cost increase [%/yr]"}]), width=12
                            ))

                        ])
                    ])
                ]),

                dbc.AccordionItem(title="About", children=[
                    dcc.Markdown('''
                    Source: https://github.com/fkuschel/LCOEcalc
                    
                    Contact:
                    - Florian Kuschel, ZBT Duisburg
                    - https://www.linkedin.com/in/florian-kuschel/
                    
                    ''')
                ]),
                # dbc.AccordionItem(title="Developer", children=[
                #     dbc.Row([dbc.Col(dbc.Button("Build: Initial Data Collect", id="bt_collect"), width=2),
                #              dbc.Col(dbc.Button("Build: Random Fill Fields", id="bt_fill"), width=2),
                #              dbc.Col(dbc.Button("Work: Update Data Collect", id="bt_update_collect"), width=2),
                #              dbc.Col(dbc.Button("Debug: Init", id="bt_init"), width=2)
                #              ]),
                #     dbc.Row([html.Pre("Nominal Calculation Done:", id="flag_nominal_calculation_done")]),
                #     dbc.Row([html.Pre("Sensitivity Calculation Done:", id="flag_sensitivity_calculation_done")]),
                #     dbc.Row([html.Pre("Build: Initial Collect Input", id="txt_build1")]),
                #     dbc.Row([html.Pre("Build: Update Collect Input", id="txt_build2")]),
                #     dbc.Row([html.Pre("Debug Calculation:", id="txt_dev_button_init")])
                # ]),
            ], always_open=True)
        ], width=12, xl=4),

        # Visualization Column
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem(title='Nominal Results', children=[
                    dbc.Row(dbc.Col(
                        dbc.Table(id="table_lcoe_nominal", striped=False, bordered=True)
                    ))
                ], ),
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
                ]),
            ], active_item=["item-0", "item-1"], always_open=True)
        ], width=12, xl=8)

    ])

], fluid=True)


# Callback Functions, app specific
# --------------------------------------------------------------
# --------------------------------------------------------------

@app.callback(
    Output("txt_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'HiPowAR', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'SOFC', 'par': ALL, 'parInfo': ALL}, 'value'),
    Output({'type': 'input', 'component': 'ICE', 'par': ALL, 'parInfo': ALL}, 'value'),

    [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(df_input["Systems"].columns[first_clm:]))], )
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

        selection_title = df_input["Systems"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Systems"].columns[first_clm:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Systems"], output=ctx.outputs_list[1:])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_Financial_Selection", "children"),
    Output({'type': 'input', 'component': 'Financials', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_Financial_{n}", "n_clicks") for n in range(len(df_input["Financial"].columns[first_clm:]))], )
def cbf_quickstart_select_financial(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Financial"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Financial"].columns[first_clm:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Financial"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NH3_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NH3', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NH3"].columns[first_clm:]))])
def cbf_quickstart_select_NH3fuel_preset(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NH3"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Fuel_NH3"].columns[first_clm:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NH3"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NG_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NG', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NG_fuel_cost_{n}", "n_clicks") for n in range(len(df_input["Fuel_NG"].columns[first_clm:]))])
def cbf_quickstart_select_NGfuel_preset(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NG"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Fuel_NG"].columns[first_clm:][0]

    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NG"], output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("flag_nominal_calculation_done", "children"),
    Output("table_lcoe_nominal", "children"),
    Input("bt_run_nominal", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': 'nominal'}, 'value'),
    prevent_initial_call=True
)
def cbf_quickstart_button_runNominalLCOE(*args):
    """
    Input:
        Button Click

    Description:
        1. Collect nominal input variables from data fields
        2. Initialize systems, prepare input-sets, perform calculations
        3. Read results and write into table

    Output:
        - Datetime to 'flag_nominal_calculation_done' textfield.
        - Single LCOE value per system to table
    """
    # 1. Collect nominal input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    data = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "nominal")

    # 3. Read results and write into table (could be reworked)
    # ------------------------------------------------------------------------------------------------------------------
    df_table = pd.DataFrame(columns=["LCOE [€/kWh]"])
    df_table.loc["System Name", "LCOE [€/kWh]"] = "LCOE [€/kWh]"
    for key, system in data.items():
        df_table.loc[key, "LCOE [€/kWh]"] = round(system.df_results.loc["nominal", "LCOE"], 3)
    df_table = df_table.reset_index(level=0)

    table = dbc.Table.from_dataframe(df_table, bordered=True, hover=True)

    logger.info('Successfull nominal calculation')

    return table.children


@app.callback(
    Output("flag_sensitivity_calculation_done", "children"),
    Output("storage", "data"),
    Output("loading-output", "children"),
    Input("bt_run_study", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True
)
def cbf_quickstart_button_runSensitivityLCOE(*args):
    """
    Input:
        Button Click

    Description:
        1. Collect all input variables from data fields
        2. Initialize systems, prepare input-sets, perform calculations
        3. Store data in dcc.storage object

    Output:
        - Datetime to 'flag_sensitivity_calculation_done' textfield.
        - system-objects, results included, to storage(s)
    """
    # 1. Collect all input variables from data fields
    # ------------------------------------------------------------------------------------------------------------------
    # Collect data of input fields in dataframe
    df = read_input_fields(ctx.states_list[0])

    # 2. Initialize systems, prepare input-sets, perform calculations
    # ------------------------------------------------------------------------------------------------------------------
    systems = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "all_minmax")

    # 3. Store data in dcc.storage object
    # -----------------------------------------------------------------------------------------------------------------
    # Create json file:
    data = store_data(systems)
    return [datetime.datetime.now(), data, None]


@app.callback(
    Output('graph_lcoe_multi_NH3', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_NH3_update(inp, state):
    """
    Todo
    """

    logger.info('Start sensitivity study')

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

    logger.info('Finished sensitivity study')

    return fig


@app.callback(
    Output('graph_lcoe_multi_NG', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_NG_update(inp, state):
    """
    Todo
    """
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
    Todo
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

    fig = make_subplots(rows=1, cols=1, shared_yaxes=True,
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

            tbred = tb.query(qs).copy()  # search for rows fullfilling query
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

    fig.update_layout(
        showlegend=False,
        boxmode='group'  # group together boxes of the different traces for each value of x
    )

    return fig


# @app.callback(
#     Output("txt_build1", "children"),
#     Input("bt_collect", "n_clicks"),
#     State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
#     prevent_initial_call=True)
# def cbf_dev_button_build_initialCollectInput(*args):
#     """
#     Creates new DataFrame / excel table with all inputfields of types defined in callback above.
#     """
#     df = build_initial_collect(ctx.states_list[0])
#     # df.to_pickle("input4.pkl")
#     df.to_excel("input4.xlsx")
#
#     return "ok"
#
#
# # # INFO: Function is commented, because there would be an output overlap. Decoment when building GUI!
# # @app.callback(
# #     Output({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
# #     Input("bt_fill", "n_clicks"),
# #     State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
# #     prevent_initial_call=True)
# # def cbf_dev_button_build_randomFillFields(*args):
# #     """
# #     Fill all fields of type 'input' with random numbers
# #     """
# #     returnvalues = build_randomfill_input_fields(ctx.states_list[0])
# #
# #     return returnvalues
#
# @app.callback(
#     Output("txt_build2", "children"), Input("bt_update_collect", "n_clicks"),
#     State({'type': 'input_HiPowAR', 'index': ALL}, 'value'),
#     State({'type': 'input_SOFC', 'index': ALL}, 'value'),
#     State({'type': 'input_ICE', 'index': ALL}, 'value'),
#     State({'type': 'input_Financials', 'index': ALL}, 'value'),
#     State({'type': 'input_Fuel_NH3', 'index': ALL}, 'value'),
#     State({'type': 'input_Fuel_NG', 'index': ALL}, 'value'),
#     prevent_initial_call=True)
# def cbf_dev_button_build_updateCollectInput(inp, *args):
#     """
#     Intention: Save new parameterset to table.
#
#     ToDo: Implement correctly!
#     """
#     df = pd.read_pickle("input4.pkl")
#     for key, val in ctx.states.items():
#         df.loc[key, inp] = val
#     df.to_pickle("input4_upd.pkl")
#     df.to_excel("input4_upd.xlsx")
#     return "ok"
#
#
# @app.callback(
#     Output("txt_dev_button_init", "children"), Input("bt_init", "n_clicks"),
#     State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
#     prevent_initial_call=True)
# def cbf_dev_button_init(inp, *args):
#     """
#     Debug parameter study
#     """
#     # Collect data of input fields in dataframe
#     df = read_input_fields(ctx.states_list[0])
#     data = multisystem_calculation(df, system_components, ["Fuel_NH3", "Fuel_NG"], "all_minmax")


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
