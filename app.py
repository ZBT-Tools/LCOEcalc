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

df_presets = pd.read_excel("input/Dash_LCOE_ConfigurationII.xlsx")
df_NH3fuel_presets = pd.read_excel("input/Dash_LCOE_NH3.xlsx")
df_NGfuel_presets = pd.read_excel("input/Dash_LCOE_NG.xlsx")

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# https://community.plotly.com/t/how-to-easily-clear-cache/7069/2
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.clear()


def input_row1(component, ident, text, type="input"):
    row = dbc.Row([
        dbc.Col(dbc.Label(text), width=6),
        dbc.Col(dbc.Input(id={'type': type, 'index': f"input_{component}_{ident}"}, type="text", size="sm"),
                width=2),
        dbc.Col(dbc.Input(id={'type': type, 'index': f"input_{component}_{ident}_min"}, type="text", disabled=True,
                          size="sm"), width=2),
        dbc.Col(dbc.Input(id={'type': type, 'index': f"input_{component}_{ident}_max"}, type="text", disabled=True,
                          size="sm"), width=2)])
    return row


def card_component_input(name: str, add_items: dict = {}):
    card_body_rows = [
        dbc.Row([
            dbc.Col(width=6),
            dbc.Col(dbc.Label("Nominal"), width=2),
            dbc.Col(dbc.Label("Min"), width=2),
            dbc.Col(dbc.Label("Max"), width=2)]),
        input_row1(component=name, ident="capex_Eur_kW", text="Capex [€/kW]"),
        input_row1(component=name, ident="opex_Eur_kWh", text="Opex (no Fuel) [€/kWh]"),
        input_row1(component=name, ident="eta_perc", text="Efficiency [%]")]
    list_additional_rows = []
    for key, val in add_items.items():
        rw = input_row1(component=name, ident=key, text=val)
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
    rows.extend([input_row1(component=component, ident=id, text=tx) for id, tx in zip(ident, text)])

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

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("HiPowAR LCOE Tool"), width=6),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(hipowar_base64), width=100)),
             dbc.Col(html.Img(src='data:image/png;base64,{}'.format(eu_base64), width=400))]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_preset", label="Select Preset",
                                                      elements=df_presets.columns[2:]), width=2),
                             dbc.Col(html.P("select...", id="txt_Preset_Selection"), width=8)]),
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_NH3_fuel_cost", label="NH3 Cost Selector",
                                                      elements=df_NH3fuel_presets.columns[2:]), width=2),
                             dbc.Col(html.P("select...", id="txt_NH3_fuel_cost_Preset_Selection"))]),
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_NG_fuel_cost", label="NG Cost Selector",
                                                      elements=df_NGfuel_presets.columns[2:]), width=2),
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
                                                         "Operating hours [hr/yr]"]), md=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Fuel Cost Settings"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(width=6),
                                    dbc.Col(dbc.Label("Nominal"), width=2),
                                    dbc.Col(dbc.Label("Min"), width=2),
                                    dbc.Col(dbc.Label("Max"), width=2)]),

                                input_row1(component="Fuel", ident="NH3_cost_EUR_per_kW", text="NH3 cost [€/kWh]",
                                           type="fuel_NH3_input"),
                                input_row1(component="Fuel", ident="NH3_costIncrease_percent_per_year",
                                           text="NH3 cost increase [%/yr]", type="fuel_NH3_input"),
                                html.Hr(),
                                input_row1(component="Fuel", ident="NG_cost_EUR_per_kW", text="NG cost [€/kWh]",
                                           type="fuel_NG_input"),
                                input_row1(component="Fuel", ident="NG_costIncrease_percent_per_year",
                                           text="NG cost increase [%/yr]", type="fuel_NG_input")
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
                             dbc.Col(dbc.Button("Export Input", id="bt_export_Input"), width=2),
                             dbc.Col(dbc.Button("Load Input", id="bt_load_Input"), width=2),
                             dbc.Col(dbc.Button("Process Input", id="bt_process_Input"), width=2)
                             ]),
                    dbc.Row([html.Pre("...", id="txt_out1")]),  # ToDo
                    dbc.Row([html.Pre("...", id="txt_out2")]),
                    dbc.Row([html.Pre("...", id="txt_out3")]),
                    dbc.Row([html.Pre("...", id="txt_out4")]),
                    dbc.Row([html.Pre("...", id="txt_out5")]),
                    dbc.Row([html.Pre("...", id="txt_out6")])
                ], title="Developer"),
            ], always_open=True)
        ]),
    ]),

], fluid=True)


# Callbacks
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------


@app.callback(
    Output("txt_Preset_Selection", "children"),
    [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(df_presets.columns[2:]))],
    prevent_initial_call=True)
def quickstart_select_preset_I(*inputs):
    selection_id = df_presets.columns[2:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    Output({'type': 'input', 'index': ALL}, 'value'), Input("txt_Preset_Selection", "children"),
    prevent_initial_call=True)
def quickstart_select_preset_II(input):
    return_list = []
    for el in ctx.outputs_list:
        el_id_index = el["id"]["index"]
        print(el_id_index)
        return_list.append(df_presets.loc[df_presets.input_name == el_id_index, input].item())
    return return_list


@app.callback(
    Output("txt_NH3_fuel_cost_Preset_Selection", "children"),
    [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(df_NH3fuel_presets.columns[2:]))],
    prevent_initial_call=True)
def quickstart_select_NH3fuel_preset_I(*input):
    selection_id = df_NH3fuel_presets.columns[2:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    Output({'type': 'fuel_NH3_input', 'index': ALL}, 'value'), Input("txt_NH3_fuel_cost_Preset_Selection", "children"),
    prevent_initial_call=True)
def quickstart_select_NH3fuel_preset_II(input):
    return_list = []
    df_NH3fuel_presets.columns[2:]
    for el in ctx.outputs_list:
        el_id_index = el["id"]["index"]
        return_list.append(df_NH3fuel_presets.loc[df_NH3fuel_presets.input_name == el_id_index, input].item())
    return return_list


@app.callback(
    Output("txt_NG_fuel_cost_Preset_Selection", "children"),
    [Input(f"dd_NG_fuel_cost_{n}", "n_clicks") for n in range(len(df_NGfuel_presets.columns[2:]))],
    prevent_initial_call=True)
def quickstart_select_NGfuel_preset_I(*input):
    selection_id = df_NGfuel_presets.columns[2:][int(ctx.triggered_id[-1])]
    return f"{selection_id}"


@app.callback(
    Output({'type': 'fuel_NG_input', 'index': ALL}, 'value'), Input("txt_NG_fuel_cost_Preset_Selection", "children"),
    prevent_initial_call=True)
def quickstart_select_NGfuel_preset_II(input):
    return_list = []
    df_NH3fuel_presets.columns[2:]
    for el in ctx.outputs_list:
        el_id_index = el["id"]["index"]
        return_list.append(df_NGfuel_presets.loc[df_NGfuel_presets.input_name == el_id_index, input].item())
    return return_list


@app.callback(
    Output("txt_out1", "children"), Input("bt_collect", "n_clicks"),
    State({'type': 'input', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def dev_button_initialCollectInput(input, states):
    df = pd.DataFrame(index=ctx.states.keys(), columns=["input_name"])
    for key, val in ctx.states.items():
        par = key.split('"')[3]
        df.loc[key, "input_name"] = par
        df.loc[key, 0] = val
    df.to_pickle("input2.pkl")
    return "ok"


@app.callback(
    Output("txt_out2", "children"), Input("bt_update_collect", "n_clicks"),
    State({'type': 'input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_updateCollectInput(input, states):
    df = pd.read_pickle("data.pkl")
    for key, val in ctx.states.items():
        df.loc[key, input] = val
    df.to_pickle("data.pkl")
    return "ok"


@app.callback(
    Output("txt_out4", "children"), Input("bt_export_Input", "n_clicks"),
    State({'type': 'fuel_NG_input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_exportInput(input, states):
    df = pd.read_pickle("input2.pkl")
    for key, val in ctx.states.items():
        df.loc[key, input] = val
    df.to_excel("test.xlsx")
    return "ok"


# @app.callback(
#     Output("txt_out6", "children"), Input("bt_process_Input", "n_clicks"),
#     State({'type': 'input', 'index': ALL}, 'value'),
#     State({'type': 'fuel_NH3_input', 'index': ALL}, 'value'),
#     State({'type': 'fuel_NG_input', 'index': ALL}, 'value'), prevent_initial_call=True)
# def dev_button_procSelection(*args):
#     # Collect all input variables and reformat to data table
#     df = pd.DataFrame(columns=["nominal", "min", "max"])
#     for sublist in ctx.states_list:
#         for el in sublist:
#             el["id"]["index"]
#     return "ok"


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
