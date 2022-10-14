"""
Dash port of Shiny iris k-means example:

https://shiny.rstudio.com/gallery/kmeans-example.html
"""
import dash
from dash import dcc
import json
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import plotly.graph_objs as go
from dash import Input, Output, dcc, html, ctx, State, MATCH, ALL
from flask_caching import Cache

df_input = pd.read_excel("input/Dash_LCOE_Configuration_v4.xlsx",
                         sheet_name=["Systems", "Financial", "Fuel_NH3", "Fuel_NG"])

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.clear()


def input_row_v1(component, ident, text):
    row = dbc.Row([
        dbc.Col(dbc.Label(text), width=6),
        dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}"}, type="text", size="sm"),
                width=2),
        dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}_min"}, type="text", disabled=True,
                          size="sm"), width=2),
        dbc.Col(dbc.Input(id={'type': f"input_{component}", 'index': f"{ident}_max"}, type="text", disabled=True,
                          size="sm"), width=2)])
    return row


def card_component_input(name: str, add_items: dict = {}):
    card_body_rows = [
        dbc.Row([
            dbc.Col(width=6),
            dbc.Col(dbc.Label("Nominal"), width=2),
            dbc.Col(dbc.Label("Min"), width=2),
            dbc.Col(dbc.Label("Max"), width=2)]),
        input_row_v1(component=name, ident="capex_Eur_kW", text="Capex [€/kW]"),
        input_row_v1(component=name, ident="opex_Eur_kWh", text="Opex (no Fuel) [€/kWh]"),
        input_row_v1(component=name, ident="eta_perc", text="Efficiency [%]")]
    list_additional_rows = []
    for key, val in add_items.items():
        rw = input_row_v1(component=name, ident=key, text=val)
        list_additional_rows.append(rw)
    card_body_rows.extend(list_additional_rows)
    card = dbc.Card([
        dbc.CardHeader(f"{name}"),
        dbc.CardBody([
            html.Div(card_body_rows)])])
    return card


def card_generic_input(component: str, header: str, ident: list, text: list):
    # Create Input rows
    rows = [dbc.Col(width=6),
            dbc.Col(dbc.Label("Nominal"), width=2),
            dbc.Col(dbc.Label("Min"), width=2),
            dbc.Col(dbc.Label("Max"), width=2)
            ]
    rows.extend([input_row_v1(component=component, ident=id, text=tx) for id, tx in zip(ident, text)])

    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody([
            html.Div([
                dbc.Row(rows)
            ])])])
    return card


def generic_dropdown(id: str, label: str, elements: list):
    dropdown = dbc.DropdownMenu(
        id=id,
        label=label,
        children=[dbc.DropdownMenuItem(el, id=f"{id}_{ct}", n_clicks=0) for ct, el in enumerate(elements)]
    )
    return dropdown


# https://community.plotly.com/t/png-image-not-showing/15713/2 # ToDo: Understand Image handling
hipowar_png = 'img/Logo_HiPowAR.png'
hipowar_base64 = base64.b64encode(open(hipowar_png, 'rb').read()).decode('ascii')
eu_png = 'img/EU_Logo.png'
eu_base64 = base64.b64encode(open(eu_png, 'rb').read()).decode('ascii')
zbt_png = 'img/logo-zbt-duisburg.png'
zbt_base64 = base64.b64encode(open(zbt_png, 'rb').read()).decode('ascii')

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("HiPowAR LCOE Tool"), width=4),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=300)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(zbt_base64), width=250))]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_preset", label="System Presets",
                                                      elements=df_input["Systems"].columns[3:]), width=2),
                             dbc.Col(html.P("select...", id="txt_Preset_Selection"), width=8)]),
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_Financial", label="Financial Presets",
                                                      elements=df_input["Financial"].columns[3:]), width=2),
                             dbc.Col(html.P("select...", id="txt_Financial_Selection"), width=8)]),
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_NH3_fuel_cost", label="NH3 Cost Selector",
                                                      elements=df_input["Fuel_NH3"].columns[3:]), width=2),
                             dbc.Col(html.P("select...", id="txt_NH3_fuel_cost_Preset_Selection"))]),
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_NG_fuel_cost", label="NG Cost Selector",
                                                      elements=df_input["Fuel_NG"].columns[3:]), width=2),
                             dbc.Col(html.P("select...", id="txt_NG_fuel_cost_Preset_Selection"))]),
                ], title="Quick Start"),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_component_input("HiPowAR"), md=4),
                        dbc.Col(card_component_input("SOFC", add_items={"stacklifetime_hr": "Stack Lifetime [hr]",
                                                                        "stackexchangecost_percCapex": "Stack Exchange Cost [% Capex"}),
                                md=4),
                        dbc.Col(card_component_input("ICE"), md=4),
                    ], )
                ], title="Energy Conversion System Definition I", ),
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

                                input_row_v1(component="Fuel_NH3", ident="cost_EUR_per_kW", text="NH3 cost [€/kWh]"),
                                input_row_v1(component="Fuel_NH3", ident="costIncrease_percent_per_year",
                                             text="NH3 cost increase [%/yr]"),
                                html.Hr(),
                                input_row_v1(component="Fuel_NG", ident="cost_EUR_per_kW", text="NG cost [€/kWh]"),
                                input_row_v1(component="Fuel_NG", ident="costIncrease_percent_per_year",
                                             text="NG cost increase [%/yr]")
                            ])
                        ]), md=4
                        ),

                    ])
                ], title="General Settings", ),
                dbc.AccordionItem([dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            figure={
                                'data': [
                                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'HiPowAR'},
                                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'NG-SOFC'},
                                    {'x': [1, 2, 3], 'y': [5, 2, 1], 'type': 'bar', 'name': 'NG-ICE'}
                                ],
                                'layout': {
                                    'title': 'Levelized Cost of Electricity Comparison'
                                }
                            }
                        )
                    ])
                ])], title="LCOE Plots"),
                dbc.AccordionItem([], title="LCOE Sensitivity Study"),
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
        ]),
    ]),

], fluid=True)


# Callbacks
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------

def fill_inputfields(input,df):
    return_lists = []
    for li in ctx.outputs_list:
        return_list = []
        for el in li:
            comp = el["id"]["type"][6:]
            par = el["id"]["index"]
            return_list.append(
                df.loc[(df.component == comp) & (df.parameter == par), input].item())
        return_lists.append(return_list)
    return return_lists




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
    prevent_initial_call=True)
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
    Input("txt_Financial_Selection", "children"),
    prevent_initial_call=True)
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
    Input("txt_NH3_fuel_cost_Preset_Selection", "children"),
    prevent_initial_call=True)
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
    Input("txt_NG_fuel_cost_Preset_Selection", "children"),
    prevent_initial_call=True)
def quickstart_select_NGfuel_preset_II(input):
    return_lists = fill_inputfields(input, df_input["Fuel_NG"])
    return return_lists

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
    Output("txt_out6", "children"), Input("bt_process_Input", "n_clicks"),
    State({'type': 'input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NH3_input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NG_input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_procSelection(*args):
    # Collect all input variables and reformat to data table
    df = pd.DataFrame(columns=["nominal", "min", "max"])

    hipowar_specific_input = {"name": "HiPowAR"}
    SOFC_specific_input = {"name": "SOFC"}
    ICE_specific_input = {"name": "ICE"}

    # Assignment of input fields to system based on name
    for el in ctx.states_list[1]:
        if el["id"]["index"].find("HiPowAR") >= 0:
            par_str = el["id"]["index"]
            par_name = par_str[par_str.find("HiPowAR") + 8:]
            hipowar_specific_input[par_name] = el["value"]
        elif el["id"]["index"].find("SOFC") >= 0:
            par_str = el["id"]["index"]
            par_name = par_str[par_str.find("SOFC") + 5:]
            SOFC_specific_input[par_name] = el["value"]
        elif el["id"]["index"].find("ICE") >= 0:
            par_str = el["id"]["index"]
            par_name = par_str[par_str.find("ICE") + 4:]
            ICE_specific_input[par_name] = el["value"]
        else:
            print(f"Non assigned element: {el['id']['index']}")

    print(hipowar_specific_input)
    print(SOFC_specific_input)
    print(ICE_specific_input)

@app.callback(
    Output("txt_out7", "children"), Input("bt_debugprint", "n_clicks"),
    State({'type': 'input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NH3_input', 'index': ALL}, 'value'),
    State({'type': 'fuel_NG_input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_debugprint(*args):
    for el in ctx.states_list[0]:
        print(el)

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
