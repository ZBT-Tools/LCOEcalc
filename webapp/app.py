""" LCOE Calculation Tool

Info
Description
    - Debugging/Developent functions and AccordeonItem are commented out.
# Ideas
    - #ToDo: Show Graphs at website startup. therefore initialize storage with default system data.

# IMPORTANT!

Code Structure:

    - Imports
    - Initialization prior to app start
    - App styling and input functions for recurring use in layout
    - App layout definition



"""
import logging.config
import os

import pandas as pd
import dash
from dash import Input, Output, dcc, html, ctx, State, ALL
from dash import DiskcacheManager, CeleryManager
import dash_bootstrap_components as dbc
import base64
import pickle
import jsonpickle
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from webapp.scripts.lcoe_calculation import multisystem_calculation
from webapp.scripts.gui_functions import fill_input_fields, read_input_fields, \
    style_generic_dropdown, \
    style_inpCard_LCOE_comp, style_inpCard_LCOE
from webapp.scripts.dash_functions import read_data, store_data

# To implement nicely
comparison_par_name = "Operation Hours [hrs/yr]"
COMP_SYS = "HiPowAR_NH3"
MODPAR = "operatinghoursyearly"

# Use Celery w/ Redis for long callbacks
#   See:
#   https://dash.plotly.com/background-callbacks
if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery

    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    celery_app.conf.update(BROKER_URL=os.environ['REDIS_URL'],
                           CELERY_RESULT_BACKEND=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache

    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

# Logging
logging.config.fileConfig('webapp/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# 1. Tool specific definitions & Initialization prior start
# ----------------------------------------------------------------------------------------------------------------------

# Note: Storage elements (dcc.Store) are defined inside app layout below

system_components = ["HiPowAR", "SOFC", "ICE"]

# Bayer; Ralf (2021): Informationsblatt CO2-Faktoren. Bundesförderung für Energie- und
# Ressourceneffizienz in der Wirtschaft - Zuschuss.
# Umweltbundesamt (2016): CO2-Emissionsfaktoren für fossile Brennstoffe.
fuel_properties = {"Fuel_NH3": {"fuel_CO2emission_tonnes_per_MWh": 0},
                   "Fuel_NG": {"fuel_CO2emission_tonnes_per_MWh": 0.02}}

# Input definition table with presets, excel table
df_input = pd.read_excel("webapp/input/Dash_LCOE_ConfigurationV5.xlsx",
                         sheet_name=["Systems", "Financial", "Fuel_NH3", "Fuel_NG"])

first_clm = 5

# Load images (issue with standard image load, due to png?!)
# Fix: https://community.plotly.com/t/png-image-not-showing/15713/2
hipowar_png = 'webapp/img/Logo_HiPowAR.png'
hipowar_base64 = base64.b64encode(open(hipowar_png, 'rb').read()).decode('ascii')
eu_png = 'webapp/img/EU_Logo.png'
eu_base64 = base64.b64encode(open(eu_png, 'rb').read()).decode('ascii')
zbt_png = 'webapp/img/ZBT_Logo_RGB_B_L-QUADRAT.jpg'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

# App initialization
app = dash.Dash(__name__)  # external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Force Plotly to clear local cache at each start
# Resolves development issue: cached data used instead of updated code
# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
# cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
# cache.clear()

# 2. App layout
# --------------------------------------------------------------------------------------------------
# Info: as proposed by dash bootstrap component guide, everything is ordered in dbc.Row's,
# containing dbc.Col's
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
# "The layout of your app should be built as a series of rows of columns. The Col component should
# always be used as an immediate child of Row and is a wrapper for your content that ensures it
# takes up the correct amount of horizontal space."
# https://getbootstrap.com/docs/5.0/utilities/spacing/

# Figure template
custom_template = {
    "layout": go.Layout(
        plot_bgcolor="#fbf5f9",
        paper_bgcolor="#fbf5f9",
    )
}
empty_template = {"layout": {"xaxis": {"visible": False},
                             "yaxis": {"visible": False},
                             "annotations": [{
                                 "text": "hi",
                                 "xref": "paper",
                                 "yref": "paper",
                                 "showarrow": False,
                                 "font": {
                                     "size": 28
                                 }
                             }],
                             "plot_bgcolor": "#fbf5f9",
                             "paper_bgcolor": "#fbf5f9"}}

empty_fig = go.Figure()
empty_fig.update_layout(template=empty_template)

app.layout = dbc.Container([

    # Storage definition
    # ---------------
    # Inputs and results are of small file size, therefore users local memory is used.
    # Limit: 'It's generally safe to store [...] 5~10MB in most desktop-only applications.'
    # https://dash.plotly.com/sharing-data-between-callbacks
    # https://dash.plotly.com/dash-core-components/store
    dcc.Store(id='nominal_storage', storage_type='memory'),
    dcc.Store(id='study_storage', storage_type='memory'),

    # Header Row with title & logos
    dbc.Row(
        [dbc.Col(html.H2(
            ["HiPowAR", html.Br(), "Electricity Generation Costs Calculation"]),
            width=12, xl={"size": 4}),
            dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100),
                    width=12, xl={"size": 2}, align="center"),
            dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=300),
                    width=12, xl=2, align="center"),
            dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250),
                    width=12, xl={"size": 2, "offset": 2}, align="center")]),
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
                        
                        
                        #### Electricity cost calculation
                        
                        Web application for comparison of electricity generation costs of different technologies with 
                        consideration of different fuels and uncertainty regarding future economic boundary condition. 
                        
                        Electricity generation costs are calculated by method of 'levelized costs of electricity', see
                        https://en.wikipedia.org/wiki/Levelized_cost_of_electricity for further details.
                        
                        #### Instruction:
                        
                        Define settings for calculation of electricity costs below. Choose from predefined system &
                        economic boundary conditions or define own study input.
                        Run calculation of nominal values or full study by clicking buttons below.

                        #### Note:
                        
                        The predefined figures are the status of the work currently in progress and are to be regarded with uncertainties. 
                        The figures are continuously updated.

                        '''),
            html.Hr(),
            dbc.Row([
                dbc.Col(dcc.Markdown('''#### Run LCOE Calculation:'''), width=6, align="center"),
                dbc.Col(dbc.Spinner(html.Div(id="loading-output"), color="success"), width=2,
                        align="center"),
                dbc.Col(dbc.Button("Nominal", id="bt_run_nominal", size="sm"), width={"size": 2}),
                dbc.Col(dbc.Button("Study", id="bt_run_study", size="sm"), width=2),

            ]),
            html.Hr(),
            dcc.Markdown('''
            ##### Settings
            '''),
            dbc.Accordion([
                dbc.AccordionItem(title="Preset Selection", children=[
                    # Menu with different drop down menus for preset selections, 'Calculate' Button
                    # Dropdown System Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_preset", label="System",

                                                       elements=df_input["Systems"].columns[
                                                                first_clm:]),
                                width=6, xxl=4),
                        dbc.Col(html.P(df_input["Systems"].columns[-1], id="txt_Preset_Selection"),
                                width=12, xxl=8)]),
                    # Dropdown Financial Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_Financial", label="Financial",
                                                       elements=df_input["Financial"].columns[
                                                                first_clm:]),
                                width=6, xxl=4),
                        dbc.Col(
                            html.P(df_input["Financial"].columns[-1], id="txt_Financial_Selection"),
                            width=12, xxl=8)]),
                    # Dropdown NH3 Fuel Cost Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_NH3_fuel_cost", label="NH3",
                                                       elements=df_input["Fuel_NH3"].columns[
                                                                first_clm:]),
                                width=6, xxl=4),
                        dbc.Col(html.P(df_input["Fuel_NH3"].columns[-1],
                                       id="txt_NH3_fuel_cost_Preset_Selection"),
                                width=12, xxl=8)]),
                    # Dropdown NG Fuel Cost Preset Selection
                    dbc.Row([
                        dbc.Col(style_generic_dropdown(id_name="dd_NG_fuel_cost", label="NG",
                                                       elements=df_input["Fuel_NG"].columns[
                                                                first_clm:]),
                                width=6, xxl=4),
                        dbc.Col(html.P(df_input["Fuel_NG"].columns[-1],
                                       id="txt_NG_fuel_cost_Preset_Selection"),
                                width=12, xxl=8)]),

                ]),

                dbc.AccordionItem(title="Energy Conversion System Settings", children=[
                    # Menu with input cards for each energy conversion system (HiPowAR, SOFC,ICE)
                    dbc.Row([
                        dbc.Col(style_inpCard_LCOE_comp(header="HiPowAR", component="HiPowAR"),
                                width=12),
                        dbc.Col(style_inpCard_LCOE_comp(header="Boiler - Baseline", component="SOFC",
                                                        # add_rows=[{"par": "stacklifetime_hr",
                                                        #            "label": "Stack Lifetime [hr]"},
                                                        #           {"par": "stackexchangecost_percCapex",
                                                        #            'label': "Stack Exchange Cost [% Capex]"}]
                                                        ),
                                width=12),

                        dbc.Col(style_inpCard_LCOE_comp(component="ICE",
                                                        header="Internal Combustion Eng."),

                                width=12)
                    ], )
                ], ),
                dbc.AccordionItem(title="Environmental Settings", children=[
                    dbc.Row([

                        dbc.Col(style_inpCard_LCOE(component="Financials",
                                                   header="Environment & Financials",

                                                   specific_row_input=[
                                                       {'par': "cost_CO2_per_tonne",
                                                        "label": "CO2 Emission Cost [€/T]"},
                                                       {'par': "CO2_costIncrease_percent_per_year",
                                                        "label": "CO2 Emission Cost Increase [%/yr]"},
                                                       {'par': "discountrate_perc",
                                                        'label': "Discount Rate [%]"},
                                                       {'par': "lifetime_yr",
                                                        'label': "Lifetime [y]"},
                                                       {'par': "operatinghoursyearly",
                                                        'label': "Operating hours [hr/yr]"},
                                                       {'par': "electricity_price",
                                                        'label': "Electricity price [€/kWh]"}]
                                                   ), width=12),
                        dbc.Col([
                            dbc.Row(dbc.Col(
                                style_inpCard_LCOE(header="NH3 Fuel", component='Fuel_NH3',
                                                   specific_row_input=[
                                                       {'par': 'fuel_cost_Eur_per_kWh',
                                                        'label': "NH3 cost [€/kWh]"},
                                                       {'par': 'fuel_costIncrease_percent_per_year',
                                                        'label': "NH3 cost increase [%/yr]"},
                                                       {'par': 'fuel_footprint_kgCO2_per_kWh',
                                                        'label': "CO2eq production footprint [kg/kWh]"},
                                                   ]),
                                width=12,
                            )),

                            dbc.Row(dbc.Col(
                                style_inpCard_LCOE(header="NG Fuel", component='Fuel_NG',
                                                   specific_row_input=[
                                                       {'par': 'fuel_cost_Eur_per_kWh',
                                                        'label': "NG cost ["
                                                                 "€/kWh]"},
                                                       {'par': 'fuel_costIncrease_percent_per_year',
                                                        'label': "NG cost increase [%/yr]"},
                                                       {'par': 'fuel_footprint_kgCO2_per_kWh',
                                                        'label': "CO2eq production footprint [kg/kWh]"}
                                                   ]),
                                width=12
                            ))

                        ])
                    ])
                ]),

                # dbc.AccordionItem(title="Developer", children=[
                #     dbc.Row([
                #         # dbc.Col(dbc.Button("Build: Initial Data Collect",
                #         #                    id="bt_collect"), width=2),
                #         # dbc.Col(dbc.Button("Build: Random Fill Fields",
                #         #                    id="bt_fill"), width=2),
                #         # dbc.Col(dbc.Button("Work: Update Data Collect",
                #         #                    id="bt_update_collect"), width=2),
                #         # dbc.Col(dbc.Button("Debug: Init", id="bt_init"), width=2)
                #         dbc.Col(dbc.Button("Debug: Save Results", id="bt_save"), width=2)
                #     ]),
                #     # dbc.Row([html.Pre("Nominal Calculation Done:",
                #     #                   id="flag_nominal_calculation_done")]),
                #     # dbc.Row([html.Pre("Sensitivity Calculation Done:",
                #     #                   id="flag_sensitivity_calculation_done")]),
                #     # dbc.Row([html.Pre("Build: Initial Collect Input", id="txt_build1")]),
                #     # dbc.Row([html.Pre("Build: Update Collect Input", id="txt_build2")]),
                #     # dbc.Row([html.Pre("Debug Calculation:", id="txt_dev_button_init")])
                # ]),
            ], always_open=True),
            html.Hr(),
            dcc.Markdown('''
            #### Contact
            https://www.linkedin.com/in/florian-kuschel/
            
            '''),
            dbc.Row([html.Div("Nominal Calculation Done:", id="flag_nominal_calculation_done",
                              style={"display": "none"})]),
            dbc.Row([html.Div("Nominal Calculation Done:", id="flag_nominal_calculation_done_2",
                              style={"display": "none"})]),
            dbc.Row([html.Div("Sensitivity Calculation Done:",
                              id="flag_sensitivity_calculation_done", style={"display": "none"})]),
            dbc.Row([html.Div("",
                              id="debug1", style={"display": "none"})]),
        ], width=12, lg=5, xxl=4),

        # Visualization Column
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem(title='Nominal Results', children=[
                    dbc.Collapse(children=[
                        dbc.Row(dbc.Col(
                            dbc.Table(id="table_lcoe_nominal", striped=False, bordered=True))),
                        dbc.Row(dcc.Graph(id='graph_pie_lcoe_nominal', figure=empty_fig)),
                        dbc.Row(dcc.Graph(id='graph_pie_emission_nominal', figure=empty_fig)),
                        dbc.Row(dcc.Graph(id='graph_yearly_cost_lcoe_nominal', figure=empty_fig)),
                        dbc.Row(
                            dcc.Graph(id='graph_yearly_revenue_lcoe_nominal', figure=empty_fig)),
                        dbc.Row(
                            dcc.Graph(id='graph_yearly_earnings_lcoe_nominal', figure=empty_fig)),
                        dbc.Row(dcc.Graph(id='graph_yearly_emission_nominal', figure=empty_fig))

                    ], id="collapse_nom", is_open=False)
                ], ),

                dbc.AccordionItem(title="LCOE Comparison Results", children=[
                    dbc.Collapse(children=[
                        dbc.Row(dbc.Col(html.H4(f"Comparisons for {comparison_par_name}"))),
                        dbc.Row(dbc.Col(
                            dbc.Table(id="table_lcoe_comparison", striped=False, bordered=True))),
                        dbc.Row(dcc.Graph(id='graph_pie_lcoe_comparison', figure=empty_fig)),
                        dbc.Row(dcc.Graph(id='graph_yearly_lcoe_comparison', figure=empty_fig))
                    ], id="collapse_comparison", is_open=False)
                ]),

                dbc.AccordionItem(title="LCOE Study Results", children=[
                    dbc.Collapse(children=[
                        dbc.Row(dcc.Graph(id='graph_lcoe_combined', figure=empty_fig)),
                        # dbc.Row([
                        #     dbc.Col([
                        #         dcc.Graph(id='graph_lcoe_multi_NH3', figure=empty_fig)
                        #     ]),
                        #     dbc.Col([
                        #         dcc.Graph(id='graph_lcoe_multi_NG', figure=empty_fig)
                        #     ])]),
                        html.Hr(),
                        dbc.Row(
                            dbc.Col([
                                dcc.Graph(id='lcoe-graph-sensitivity', figure=empty_fig)])
                        )], id="collapse_study", is_open=False)])
                ,
            ], active_item=["item-0", "item-1"], always_open=True)  #
        ], width=12, lg=7, xxl=8)

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

    [Input(f"dd_preset_{n}", "n_clicks") for n in
     range(len(df_input["Systems"].columns[first_clm:]))], )
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

    return_lists = fill_input_fields(selection_title, df=df_input["Systems"],
                                     output=ctx.outputs_list[1:])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_Financial_Selection", "children"),
    Output({'type': 'input', 'component': 'Financials', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_Financial_{n}", "n_clicks") for n in
     range(len(df_input["Financial"].columns[first_clm:]))], )
def cbf_quickstart_select_financial(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Financial"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Financial"].columns[first_clm:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Financial"],
                                     output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NH3_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NH3', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in
     range(len(df_input["Fuel_NH3"].columns[first_clm:]))])
def cbf_quickstart_select_NH3fuel_preset(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NH3"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Fuel_NH3"].columns[first_clm:][-1]

    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NH3"],
                                     output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("txt_NG_fuel_cost_Preset_Selection", "children"),
    Output({'type': 'input', 'component': 'Fuel_NG', 'par': ALL, 'parInfo': ALL}, 'value'),
    [Input(f"dd_NG_fuel_cost_{n}", "n_clicks") for n in
     range(len(df_input["Fuel_NG"].columns[first_clm:]))])
def cbf_quickstart_select_NGfuel_preset(*inputs):
    """
    Same as for cbf_quickstart_select_system_preset
    """
    try:
        selection_title = df_input["Fuel_NG"].columns[first_clm:][int(ctx.triggered_id[-1])]
    except TypeError:
        # At app initialization, callback is executed witout trigger id. Select newest definition
        selection_title = df_input["Fuel_NG"].columns[first_clm:][0]

    return_lists = fill_input_fields(selection_title, df=df_input["Fuel_NG"],
                                     output=ctx.outputs_list[1])

    output = [selection_title]
    output.extend(return_lists)

    return output


@app.callback(
    Output("flag_nominal_calculation_done", "children"),
    Output("nominal_storage", "data"),
    Output("collapse_nom", "is_open"),
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
    data = multisystem_calculation(df, system_names=system_components,
                                   fuel_names=["Fuel_NH3", "Fuel_NG"], fuel_prop=fuel_properties,
                                   mode="nominal")

    stor_data = store_data(data)

    logger.info('Successfull nominal calculation')

    return datetime.datetime.now(), stor_data, True


@app.callback(
    Output("flag_sensitivity_calculation_done", "children"),
    Output("study_storage", "data"),
    Output("loading-output", "children"),
    Output("collapse_study", "is_open"),
    Output("collapse_comparison", "is_open"),
    Input("bt_run_study", "n_clicks"),
    State({'type': 'input', 'component': ALL, 'par': ALL, 'parInfo': ALL}, 'value'),
    prevent_initial_call=True,
    background=True,
    manager=background_callback_manager
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
    # nominal_system = multisystem_calculation(df, system_names=system_components,
    #                                          fuel_names=["Fuel_NH3", "Fuel_NG"],
    #                                          fuel_prop=fuel_properties, mode="nominal")

    systems = multisystem_calculation(df, system_names=system_components,
                                      fuel_names=["Fuel_NH3", "Fuel_NG"],
                                      fuel_prop=fuel_properties, mode="all_minmax")

    # 3. Store data in dcc.storage object
    # -----------------------------------------------------------------------------------------------------------------
    # Create json file:
    # nominal_data = store_data(nominal_system)
    study_data = store_data(systems)
    return [datetime.datetime.now(), study_data, None, True, True]


@app.callback(
    Output("table_lcoe_nominal", "children"),
    Input("flag_nominal_calculation_done", "children"),
    # Input("flag_nominal_calculation_done_2", "children"),
    State('nominal_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeNominalResults_table_update(inp, state):
    """
    Update table
    """
    data = read_data(state)

    # Write into table (could be reworked)
    # ----------------------------------------------------------------------------------------------
    df_table = pd.DataFrame(
        columns=["System Name", "LCOE [€/kWh], Ammonia", "LCOE [€/kWh], Natural Gas"])
    for key, system in data.items():
        systemname = key.split("_")[0]
        df_table.loc[systemname, "System Name"] = key.split("_")[0]
        if key.split("_")[1] == "NH3":
            df_table.loc[systemname, "LCOE [€/kWh], Ammonia"] = \
                f'{system.df_results.loc["nominal", "LCOE"]:9.2f}'

        else:
            df_table.loc[systemname, "LCOE [€/kWh], Natural Gas"] = \
                f'{system.df_results.loc["nominal", "LCOE"]:9.2f}'

    table = dbc.Table.from_dataframe(df_table, bordered=True, hover=True, index=False, header=True)

    return table.children


@app.callback(
    Output("graph_pie_lcoe_nominal", "figure"),
    Input("flag_nominal_calculation_done", "children"),
    State('nominal_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeNominalResults_piechart_update(inp, state):
    """
    Update table
    """
    data = read_data(state)

    labels = ["Capex", "Opex", "Fuel", "Emissions"]
    HiPowAR_data = [sum(data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Investment_fin),
                    sum(data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.OM_fin),
                    sum(data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Fuel_fin),
                    sum(data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Cost_fin)]
    SOFC_data = [sum(data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Investment_fin),
                 sum(data["SOFC_NH3"].df_results.LCOE_detailed.nominal.OM_fin),
                 sum(data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Fuel_fin),
                 sum(data["SOFC_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Cost_fin)]
    ICE_data = [sum(data["ICE_NH3"].df_results.LCOE_detailed.nominal.Investment_fin),
                sum(data["ICE_NH3"].df_results.LCOE_detailed.nominal.OM_fin),
                sum(data["ICE_NH3"].df_results.LCOE_detailed.nominal.Fuel_fin),
                sum(data["ICE_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Cost_fin)]

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=HiPowAR_data, name="HiPowAR system"),
                  1, 1)
    fig.add_trace(go.Pie(labels=labels, values=SOFC_data, name="SOFC system"),
                  1, 2)
    fig.add_trace(go.Pie(labels=labels, values=ICE_data, name="ICE system"),
                  1, 3)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name", textinfo='label')

    fig.update_layout(
        title_text="Cost distribution, Net Present Values (Ammonia fueled systems)",
        showlegend=False,
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='HiPowAR', x=0.10, y=-0.1, font_size=15, showarrow=False),
                     dict(text='SOFC', x=0.5, y=-0.1, font_size=15, showarrow=False),
                     dict(text='ICE', x=0.872, y=-0.1, font_size=15, showarrow=False)],

        template=custom_template,
        autosize=False,
        # width=500,
        height=400
    )

    return fig


# @app.callback(
#     Output("graph_yearly_cost_lcoe_nominal", "figure"),
#     Input("flag_nominal_calculation_done", "children"),
#     State('nominal_storage', 'data'),
#     prevent_initial_call=True)
# def cbf_lcoeNominalResults_yearly_cost_chart_update(inp, state):
#     """
#     Update yearly cost  line plot
#     """
#     data = read_data(state)
#
#     labels = ["Capex", "Opex", "Fuel"]
#     years = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.index
#     HiPowAR_NH3_data = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
#     SOFC_NH3_data = data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
#     ICE_NH3_data = data["ICE_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
#     SOFC_NG_data = data["SOFC_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum
#     ICE_NG_data = data["ICE_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum
#
#     # Create traces
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=years, y=HiPowAR_NH3_data,
#                              mode='lines+markers',
#                              name='HiPowAR'))
#     fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data,
#                              mode='lines+markers',
#                              name='SOFC'))
#     fig.add_trace(go.Scatter(x=years, y=ICE_NH3_data,
#                              mode='lines+markers',
#                              name='ICE'))
#     fig.add_trace(go.Scatter(x=years, y=SOFC_NG_data,
#                              mode='lines+markers',
#                              line=dict(dash='dash'),
#                              name='SOFC NG'))
#     fig.add_trace(go.Scatter(x=years, y=ICE_NG_data,
#                              mode='lines+markers',
#                              line=dict(dash='dash'),
#                              name='ICE NG'))
#
#     fig.update_layout(
#         title_text="Cost Development",
#         # Add annotations in the center of the donut pies.
#         # annotations=[dict(text='HiPowAR', x=0.11, y=0.5, font_size=15, showarrow=False),
#         #              dict(text='SOFC', x=0.5, y=0.5, font_size=15, showarrow=False),
#         #              dict(text='ICE', x=0.875, y=0.5, font_size=15, showarrow=False)],
#
#         template=custom_template,
#         autosize=False,
#         # width=500,
#         height=500,
#         xaxis_title='Year',
#         yaxis_title='Cumulated Cost [€]')
#
#     return fig


@app.callback(
    Output("graph_yearly_revenue_lcoe_nominal", "figure"),
    Input("flag_nominal_calculation_done", "children"),
    State('nominal_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeNominalResults_yearly_revenue_chart_update(inp, state):
    """
    Update yearly cost vs . saving/revenue line plot
    """
    data = read_data(state)

    years = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.index
    HiPowAR_NH3_data_revenue = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_fin_cum
    SOFC_NH3_data_revenue = data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_fin_cum
    ICE_NH3_data_revenue = data["ICE_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_fin_cum
    SOFC_NG_data_revenue = data["SOFC_NG"].df_results.LCOE_detailed.nominal.Power_Revenue_fin_cum
    ICE_NG_data_revenue = data["ICE_NG"].df_results.LCOE_detailed.nominal.Power_Revenue_fin_cum

    HiPowAR_NH3_data_cost = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_fin_cum
    SOFC_NH3_data_cost = data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_fin_cum
    ICE_NH3_data_cost = data["ICE_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_fin_cum
    SOFC_NG_data_cost = data["SOFC_NG"].df_results.LCOE_detailed.nominal.Cost_combined_fin_cum
    ICE_NG_data_cost = data["ICE_NG"].df_results.LCOE_detailed.nominal.Cost_combined_fin_cum

    delta_cost = HiPowAR_NH3_data_cost - SOFC_NH3_data_cost
    additional_revenue = HiPowAR_NH3_data_revenue - SOFC_NH3_data_revenue

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=delta_cost,
                             mode='lines+markers',
                             name='C post_exchange',
                             line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=years, y=additional_revenue,
                             mode='lines+markers',
                             name='S post_exchange',
                             line=dict(color='red', width=1,
                                       )#dash='dash')
                             ))
    fig.add_trace(go.Scatter(x=years, y=additional_revenue - delta_cost,
                             mode='lines+markers',
                             name='R post_exchange',
                             line=dict(color='green', width=1,
                                       )#dash='dot')
                             ))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data_revenue,
    #                          mode='lines+markers',
    #                          name='Boiler revenue|saving',
    #                          line=dict(color='firebrick', width=1)
    #                          ))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data_cost,
    #                          mode='lines+markers',
    #                          name='Boiler cost',
    #                          line=dict(color='firebrick', width=1,
    #                                    dash='dash')
    #                          ))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data_revenue-SOFC_NH3_data_cost,
    #                          mode='lines+markers',
    #                          name='Boiler revenue|saving - cost',
    #                          line=dict(color='firebrick', width=1,
    #                                    dash='dash')
    #                          ))
    # fig.add_trace(go.Scatter(x=years, y=ICE_NH3_data,
    #                          mode='lines+markers',
    #                          name='ICE'))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NG_data,
    #                          mode='lines+markers',
    #                          line=dict(dash='dash'),
    #                          name='SOFC NG'))
    # fig.add_trace(go.Scatter(x=years, y=ICE_NG_data,
    #                          mode='lines+markers',
    #                          line=dict(dash='dash'),
    #                          name='ICE NG'))


    fig.update_layout(
        title_text="Cumulative Savings and Costs of HiPowAR exchange",
        # Add annotations in the center of the donut pies.
        # annotations=[dict(text='HiPowAR', x=0.11, y=0.5, font_size=15, showarrow=False),
        #              dict(text='SOFC', x=0.5, y=0.5, font_size=15, showarrow=False),
        #              dict(text='ICE', x=0.875, y=0.5, font_size=15, showarrow=False)],

        template="plotly_white",
        #template=custom_template,
        autosize=False,
        # width=500,
        height=500,

        xaxis_title='Year',
        yaxis_title='Savings / Costs [€]')

    return fig


# @app.callback(
#     Output("graph_yearly_earnings_lcoe_nominal", "figure"),
#     Input("flag_nominal_calculation_done", "children"),
#     State('nominal_storage', 'data'),
#     prevent_initial_call=True)
# def cbf_lcoeNominalResults_yearly_earnings_chart_update(inp, state):
#     """
#     Update yearly earning line plot
#     """
#     data = read_data(state)
#
#     years = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.index
#     HiPowAR_NH3_data = (data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_cum -
#                         data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum)
#     SOFC_NH3_data = (data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_cum -
#                      data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum)
#     ICE_NH3_data = (data["ICE_NH3"].df_results.LCOE_detailed.nominal.Power_Revenue_cum -
#                     data["ICE_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum)
#     SOFC_NG_data = (data["SOFC_NG"].df_results.LCOE_detailed.nominal.Power_Revenue_cum -
#                     data["SOFC_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum)
#     ICE_NG_data = (data["ICE_NG"].df_results.LCOE_detailed.nominal.Power_Revenue_cum -
#                    data["ICE_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum)
#
#     # Create traces
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=years, y=HiPowAR_NH3_data,
#                              mode='lines+markers',
#                              name='HiPowAR'))
#     fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data,
#                              mode='lines+markers',
#                              name='Boiler'))
#     # fig.add_trace(go.Scatter(x=years, y=ICE_NH3_data,
#     #                          mode='lines+markers',
#     #                          name='ICE'))
#     # fig.add_trace(go.Scatter(x=years, y=SOFC_NG_data,
#     #                          mode='lines+markers',
#     #                          line=dict(dash='dash'),
#     #                          name='SOFC NG'))
#     # fig.add_trace(go.Scatter(x=years, y=ICE_NG_data,
#     #                          mode='lines+markers',
#     #                          line=dict(dash='dash'),
#     #                          name='ICE NG'))
#
#     fig.update_layout(
#         title_text="Revenue|Saving - Cost Development",
#         # Add annotations in the center of the donut pies.
#         # annotations=[dict(text='HiPowAR', x=0.11, y=0.5, font_size=15, showarrow=False),
#         #              dict(text='SOFC', x=0.5, y=0.5, font_size=15, showarrow=False),
#         #              dict(text='ICE', x=0.875, y=0.5, font_size=15, showarrow=False)],
#
#         template=custom_template,
#         autosize=False,
#         # width=500,
#         height=500,
#         xaxis_title='Year',
#         yaxis_title='Revenue|Saving - Cost [€]')
#
#     return fig


@app.callback(
    Output("graph_yearly_emission_nominal", "figure"),
    Input("flag_nominal_calculation_done", "children"),
    State('nominal_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeNominalEmissionResults_yearly_chart_update(inp, state):
    """
    Update yearly emission line plot
    """
    data = read_data(state)

    labels = ["Capex", "Opex", "Fuel"]
    years = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.index
    HiPowAR_NH3_data = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Tonnes_cum
    SOFC_NH3_data = data["SOFC_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Tonnes_cum
    ICE_NH3_data = data["ICE_NH3"].df_results.LCOE_detailed.nominal.CO2_Emission_Tonnes_cum
    SOFC_NG_data = data["SOFC_NG"].df_results.LCOE_detailed.nominal.CO2_Emission_Tonnes_cum
    ICE_NG_data = data["ICE_NG"].df_results.LCOE_detailed.nominal.CO2_Emission_Tonnes_cum

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=HiPowAR_NH3_data,
                             mode='lines+markers',
                             name='HiPowAR'))
    fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data,
                             mode='lines+markers',
                             name='SOFC'))
    fig.add_trace(go.Scatter(x=years, y=ICE_NH3_data,
                             mode='lines+markers',
                             name='ICE'))
    fig.add_trace(go.Scatter(x=years, y=SOFC_NG_data,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='SOFC NG'))
    fig.add_trace(go.Scatter(x=years, y=ICE_NG_data,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='ICE NG'))

    fig.update_layout(
        title_text="Emission Development",
        # Add annotations in the center of the donut pies.
        template=custom_template,
        autosize=False,
        # width=500,
        height=500,
        xaxis_title='Year',
        yaxis_title='Cumulated Emissions, CO² eq. [T]')

    return fig


@app.callback(
    Output("table_lcoe_comparison", "children"),
    Input("flag_sensitivity_calculation_done", "children"),
    State('study_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeComparisonResults_table_update(inp, state):
    """
    Update table
    """
    data = read_data(state)

    # Get correct results out of study data
    system = COMP_SYS
    modpar = MODPAR
    tb = data[system].df_results.copy()
    tb = tb.apply(pd.to_numeric, errors='ignore')  # Todo relocate

    variation_pars = tb.columns.drop(
        ["size_kW", "LCOE", "name", "fuel_name", "fuel_CO2emission_tonnes_per_MWh",
         "LCOE_detailed"])

    # Build new dataframe for plotting
    result_df = pd.DataFrame(columns=tb.columns)
    result_df.loc["nominal"] = tb.loc["nominal"]  # Always include nominal calculation
    result_df_temp = result_df.copy()

    # Create query string:
    qs = ""
    # Next rows create query:
    # Find all result rows, where all other values beside modpar are nomial
    cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in
            variation_pars.drop(modpar)]
    for c in cond:
        qs = qs + c + " & "
    qs = qs[:-3]  # remove last  " & "

    tbred = tb.query(qs).copy()  # search for rows fullfilling query
    # In case modpar has no variation (e.g. fuel cost increase is set as [0,0,0], all values are the same.
    # Thus following rows could include "nominal" set again. This needs to be prevented.
    tbred.drop(index="nominal", inplace=True)
    rw = tbred.nsmallest(1, modpar)  # find smallest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])
    rw = tbred.nlargest(1,
                        modpar)  # find largest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])

    result_df.drop_duplicates(keep='first', subset=result_df.columns.difference(['LCOE_detailed']),
                              inplace=True)

    # Write into table (could be reworked)
    # ----------------------------------------------------------------------------------------------
    df_table = pd.DataFrame(
        columns=["Configuration", "LCOE [€/kWh]"])

    names = ["nominal", "min", "max"]
    for (index, row), name in zip(result_df.iterrows(), names):
        systemname = name
        df_table.loc[systemname, "Configuration"] = name
        df_table.loc[systemname, "LCOE [€/kWh]"] = \
            f'{row.LCOE:9.2f}'

    table = dbc.Table.from_dataframe(df_table, bordered=True, hover=True, index=False, header=True)

    return table.children


@app.callback(
    Output("graph_pie_lcoe_comparison", "figure"),
    Input("flag_sensitivity_calculation_done", "children"),
    State('study_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeComparisonResults_piechart_update(inp, state):
    """
    Update yearly line plot for comparison
    """
    data = read_data(state)

    # Get correct results out of study data
    system = COMP_SYS
    modpar = MODPAR
    tb = data[system].df_results.copy()
    tb = tb.apply(pd.to_numeric, errors='ignore')  # Todo relocate

    variation_pars = tb.columns.drop(
        ["size_kW", "LCOE", "name", "fuel_name", "fuel_CO2emission_tonnes_per_MWh",
         "LCOE_detailed"])

    # Build new dataframe for plotting
    result_df = pd.DataFrame(columns=tb.columns)
    result_df.loc["nominal"] = tb.loc["nominal"]  # Always include nominal calculation
    result_df_temp = result_df.copy()

    # Create query string:
    qs = ""
    # Next rows create query:
    # Find all result rows, where all other values beside modpar are nomial
    cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in
            variation_pars.drop(modpar)]
    for c in cond:
        qs = qs + c + " & "
    qs = qs[:-3]  # remove last  " & "

    tbred = tb.query(qs).copy()  # search for rows fullfilling query
    # In case modpar has no variation (e.g. fuel cost increase is set as [0,0,0], all values are the same.
    # Thus following rows could include "nominal" set again. This needs to be prevented.
    tbred.drop(index="nominal", inplace=True)
    rw = tbred.nsmallest(1, modpar)  # find smallest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])
    rw = tbred.nlargest(1,
                        modpar)  # find largest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])

    result_df.drop_duplicates(keep='first', subset=result_df.columns.difference(['LCOE_detailed']),
                              inplace=True)

    labels = ["Capex", "Opex", "Fuel"]
    data = []
    for index, row in result_df.iterrows():
        data.append([sum(row["LCOE_detailed"].Investment_fin),
                     sum(row["LCOE_detailed"].OM_fin),
                     sum(row["LCOE_detailed"].Fuel_fin) + sum(
                         row["LCOE_detailed"].CO2_Emission_Cost_fin)])

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=data[0], name="Nominal"),
                  1, 1)
    fig.add_trace(go.Pie(labels=labels, values=data[1], name="Min"),
                  1, 2)
    fig.add_trace(go.Pie(labels=labels, values=data[2], name="Max"),
                  1, 3)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name", textinfo='label')

    fig.update_layout(
        title_text="Cost distribution, Net Present Values",
        showlegend=False,
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Nominal', x=0.10, y=-0.1, font_size=15, showarrow=False),
                     dict(text='Min', x=0.5, y=-0.1, font_size=15, showarrow=False),
                     dict(text='Max', x=0.872, y=-0.1, font_size=15, showarrow=False)],

        template=custom_template,
        autosize=False,
        # width=500,
        height=400
    )

    return fig


@app.callback(
    Output("graph_yearly_lcoe_comparison", "figure"),
    Input("flag_sensitivity_calculation_done", "children"),
    State('study_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeComparisonResults_yearly_chart_update(inp, state):
    """
    Update yearly line plot for comparison
    """
    data = read_data(state)

    # Get correct results out of study data
    system = COMP_SYS
    modpar = MODPAR
    tb = data[system].df_results.copy()
    tb = tb.apply(pd.to_numeric, errors='ignore')  # Todo relocate

    variation_pars = tb.columns.drop(
        ["size_kW", "LCOE", "name", "fuel_name", "fuel_CO2emission_tonnes_per_MWh",
         "LCOE_detailed"])

    # Build new dataframe for plotting
    result_df = pd.DataFrame(columns=tb.columns)
    result_df.loc["nominal"] = tb.loc["nominal"]  # Always include nominal calculation
    result_df_temp = result_df.copy()

    # Create query string:
    qs = ""
    # Next rows create query:
    # Find all result rows, where all other values beside modpar are nomial
    cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in
            variation_pars.drop(modpar)]
    for c in cond:
        qs = qs + c + " & "
    qs = qs[:-3]  # remove last  " & "

    tbred = tb.query(qs).copy()  # search for rows fullfilling query
    # In case modpar has no variation (e.g. fuel cost increase is set as [0,0,0], all values are the same.
    # Thus following rows could include "nominal" set again. This needs to be prevented.
    tbred.drop(index="nominal", inplace=True)
    rw = tbred.nsmallest(1, modpar)  # find smallest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])
    rw = tbred.nlargest(1,
                        modpar)  # find largest value of modpar for all results and add
    # to result_df
    rw["modpar"] = modpar
    result_df = pd.concat([result_df, rw])

    result_df.drop_duplicates(keep='first', subset=result_df.columns.difference(['LCOE_detailed']),
                              inplace=True)

    # Plotting
    years = result_df.loc["nominal", "LCOE_detailed"].index
    sys_data = result_df.loc["nominal", "LCOE_detailed"].Cost_combined_cum

    fig = go.Figure()
    names = ["nominal", "min", "max"]
    for (index, row), name in zip(result_df.iterrows(), names):
        sys_data = row["LCOE_detailed"].Cost_combined_cum
        fig.add_trace(go.Scatter(x=years, y=sys_data,
                                 mode='lines+markers',
                                 name=name))

    fig.update_layout(
        title_text="Cost Development",
        # Add annotations in the center of the donut pies.
        # annotations=[dict(text='HiPowAR', x=0.11, y=0.5, font_size=15, showarrow=False),
        #              dict(text='SOFC', x=0.5, y=0.5, font_size=15, showarrow=False),
        #              dict(text='ICE', x=0.875, y=0.5, font_size=15, showarrow=False)],

        template=custom_template,
        autosize=False,
        # width=500,
        height=500,
        xaxis_title='Year',
        yaxis_title='Cumulated Cost [€]')

    return fig

    #
    #
    # labels = ["Capex", "Opex", "Fuel"]
    # years = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.index
    # HiPowAR_NH3_data = data["HiPowAR_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
    # SOFC_NH3_data = data["SOFC_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
    # ICE_NH3_data = data["ICE_NH3"].df_results.LCOE_detailed.nominal.Cost_combined_cum
    # SOFC_NG_data = data["SOFC_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum
    # ICE_NG_data = data["ICE_NG"].df_results.LCOE_detailed.nominal.Cost_combined_cum
    #
    # # Create traces
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=years, y=HiPowAR_NH3_data,
    #                          mode='lines+markers',
    #                          name='HiPowAR'))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NH3_data,
    #                          mode='lines+markers',
    #                          name='SOFC'))
    # fig.add_trace(go.Scatter(x=years, y=ICE_NH3_data,
    #                          mode='lines+markers',
    #                          name='ICE'))
    # fig.add_trace(go.Scatter(x=years, y=SOFC_NG_data,
    #                          mode='lines+markers',
    #                          line=dict(dash='dash'),
    #                          name='SOFC NG'))
    # fig.add_trace(go.Scatter(x=years, y=ICE_NG_data,
    #                          mode='lines+markers',
    #                          line=dict(dash='dash'),
    #                          name='ICE NG'))
    #
    # fig.update_layout(
    #     title_text="Cost Development",
    #     # Add annotations in the center of the donut pies.
    #     # annotations=[dict(text='HiPowAR', x=0.11, y=0.5, font_size=15, showarrow=False),
    #     #              dict(text='SOFC', x=0.5, y=0.5, font_size=15, showarrow=False),
    #     #              dict(text='ICE', x=0.875, y=0.5, font_size=15, showarrow=False)],
    #
    #     template=custom_template,
    #     autosize=False,
    #     # width=500,
    #     height=500,
    #     xaxis_title='Year',
    #     yaxis_title='Cumulated Cost [€]')
    #
    # return fig


# @app.callback(
#     Output("table_lcoe_comparison", "children"),
#     Input("flag_sensitivity_calculation_done", "children"),
#     State('nominal_storage', 'data'),
#     prevent_initial_call=True)
# def cbf_lcoeComparisonResults_table_update(inp, state):
#     """
#     Update table
#     """
#     data = read_data(state)
#
#     # Write into table (could be reworked)
#     # ----------------------------------------------------------------------------------------------
#     df_table = pd.DataFrame(
#         columns=["System Name", "LCOE [€/kWh], Ammonia", "LCOE [€/kWh], Natural Gas"])
#     for key, system in data.items():
#         systemname = key.split("_")[0]
#         df_table.loc[systemname, "System Name"] = key.split("_")[0]
#         if key.split("_")[1] == "NH3":
#             df_table.loc[systemname, "LCOE [€/kWh], Ammonia"] = \
#                 f'{system.df_results.loc["nominal", "LCOE"]:9.2f}'
#
#         else:
#             df_table.loc[systemname, "LCOE [€/kWh], Natural Gas"] = \
#                 f'{system.df_results.loc["nominal", "LCOE"]:9.2f}'
#
#     table = dbc.Table.from_dataframe(df_table, bordered=True, hover=True, index=False, header=True)
#
#     return table.children

@app.callback(
    Output('graph_lcoe_combined', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('study_storage', 'data'),
    prevent_initial_call=True)
def cbf_lcoeStudyResults_plot_update(inp, state):
    """
    ...
    """

    # Read results from storage
    systems = read_data(state)

    # Simple LCOE Comparison Plot
    y0 = systems["HiPowAR_NH3"].df_results["LCOE"]
    y1 = systems["SOFC_NH3"].df_results["LCOE"]
    y2 = systems["ICE_NH3"].df_results["LCOE"]
    y3 = systems["SOFC_NG"].df_results["LCOE"]
    y4 = systems["ICE_NG"].df_results["LCOE"]
    fig = go.Figure()

    fig.add_trace(go.Box(y=y0, name='HiPowAR',
                         boxpoints=False,
                         marker=dict(color='rgb(160,7,97)'),
                         line=dict(color='rgb(160,7,97)'),  # rgb(31,148,175),

                         ))
    fig.add_trace(go.Box(y=y1, name='SOFC',
                         marker=dict(color='lightseagreen'), boxpoints=False))
    fig.add_trace(go.Box(y=y2, name='ICE',
                         marker=dict(color='lightskyblue'), boxpoints=False))
    fig.add_trace(go.Box(y=y3, name='SOFC, NG',
                         marker=dict(color='lightseagreen'), boxpoints=False))
    fig.add_trace(go.Box(y=y4, name='ICE, NG',
                         marker=dict(color='lightskyblue'), boxpoints=False))

    fig.update_layout(
        title="Levelized Cost of Electricity - Green Ammonia ",
        # xaxis_title="",
        yaxis_title="LCOE [€/kWh]",
        template=custom_template)

    fig.add_vline(x=2.4, line_width=3, line_dash="dash", line_color="green")

    # add annotation
    fig.add_annotation(dict(font=dict(size=15),  # ,color='yellow',
                            x=0.6,
                            y=-0.2,
                            showarrow=False,
                            text="Natural Gas",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))

    return fig


@app.callback(
    Output('lcoe-graph-sensitivity', 'figure'),
    Input("flag_sensitivity_calculation_done", "children"),
    State('study_storage', 'data'),
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

    colordict = {"HiPowAR_NH3": 'rgb(160,7,97)', "SOFC_NH3": 'lightseagreen',
                 "ICE_NH3": 'lightskyblue'}

    fig = make_subplots(rows=1, cols=1, shared_yaxes=True,
                        # x_title='Your master x-title',
                        y_title='LCOE [€/kWh]',
                        subplot_titles=(
                            'System Sensitivity (Ammonia Systems)', 'Environment Sensitivity'))

    for system in ["HiPowAR_NH3", "SOFC_NH3", "ICE_NH3"]:

        tb = systems[system].df_results.copy()
        tb = tb.apply(pd.to_numeric, errors='ignore')  # Todo relocate

        # Create first plot with only system parameters, identified by "p".

        variation_pars = tb.columns.drop(
            ["size_kW", "LCOE", "name", "fuel_name", "fuel_CO2emission_tonnes_per_MWh",
             "CO2_costIncrease_percent_per_year",
             "cost_CO2_per_tonne", "LCOE_detailed"])

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
            cond = [f"{parm} == {result_df_temp.loc['nominal', parm]}" for parm in
                    variation_pars.drop(modpar)]
            for c in cond:
                qs = qs + c + " & "
            qs = qs[:-3]  # remove last  " & "

            tbred = tb.query(qs).copy()  # search for rows fullfilling query
            # In case modpar has no variation (e.g. fuel cost increase is set as [0,0,0], all values are the same.
            # Thus following rows could include "nominal" set again. This needs to be prevented.
            tbred.drop(index="nominal", inplace=True)
            rw = tbred.nsmallest(1,
                                 modpar)  # find smallest value of modpar for all results and add
            # to result_df
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])
            rw = tbred.nlargest(1,
                                modpar)  # find largest value of modpar for all results and add
            # to result_df
            rw["modpar"] = modpar
            result_df = pd.concat([result_df, rw])

        result_df.loc[:, "diff"] = result_df["LCOE"] - result_df.loc[
            "nominal", "LCOE"]  # Calculate difference to nominal

        result_df.drop_duplicates(keep='first',
                                  subset=result_df.columns.difference(['LCOE_detailed']),
                                  inplace=True)

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
        boxmode='group',  # group together boxes of the different traces for each value of x
        template=custom_template
    )

    return fig


# @app.callback(
#     Output("debug1", "children"),
#     Input("bt_save", "n_clicks"),
#     State('storage', 'data'),
#     prevent_initial_call=True)
# def cbf_dev_button_save_study(clicks, data):
#     """
#     Save study results
#     """
#     file = store_data(data)
#     with open('input/results.pickle', 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
    app.run_server(debug=True, port=8000)
