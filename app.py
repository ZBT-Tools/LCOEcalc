"""
Dash port of Shiny iris k-means example:

https://shiny.rstudio.com/gallery/kmeans-example.html
"""
import dash
import json
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html, ctx, State, MATCH, ALL

# Definitions ToDo: Find best position
preset_Cases = ["100kW base case, Ver. 09/22",
                "1MW  base case, Ver. 09/22"]

NH3_fuel_cost_Cases = ["NH3 Today",
                       "IRENA Outlook Green NH3 2030",
                       "IRENA Outlook Green NH3 2040",
                       ]
NG_fuel_cost_Cases = ["NG Today",
                      "IRENA Outlook 2030",
                      "IRENA Outlook 2040",
                      ]

df_definitions = pd.read_excel("Dash_LCOE_Configuration.xlsx")


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def input_row1(component, ident, text):
    row = dbc.Row([
        dbc.Col(dbc.Label(text), width=6),
        dbc.Col(dbc.Input(id={'type': 'input', 'index': f"input_{component}_{ident}"}, type="text", size="sm"),
                width=2),
        dbc.Col(dbc.Input(id={'type': 'input', 'index': f"input_{component}_{ident}_min"}, type="text", disabled=True,
                          size="sm"), width=2),
        dbc.Col(dbc.Input(id={'type': 'input', 'index': f"input_{component}_{ident}_max"}, type="text", disabled=True,
                          size="sm"), width=2)])
    return row


def card_component_input(name: str):
    card = dbc.Card([
        dbc.CardHeader(f"{name}"),
        dbc.CardBody([
            html.Div([
                dbc.Row([
                    dbc.Col(width=6),
                    dbc.Col(dbc.Label("Nominal"), width=2),
                    dbc.Col(dbc.Label("Min"), width=2),
                    dbc.Col(dbc.Label("Max"), width=2)]),
                input_row1(component=name, ident="capex", text="Capex [€/kW]"),
                input_row1(component=name, ident="opex", text="Opex (no Fuel) [€/kW]"),
                input_row1(component=name, ident="eta", text="Efficiency [%]"),
            ])])])
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


app.layout = dbc.Container([
    html.H1("HiPowAR LCOE Tool"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([dbc.Col(generic_dropdown(id="dd_preset", label="Select Preset", elements=preset_Cases),
                                     width=2),
                             dbc.Col(html.P(id="txt_Preset_Selection"), width=8)]),
                ], title="Quick Start"),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_component_input("HiPowAR"), md=4),
                        dbc.Col(card_component_input("SOFC"), md=4),
                        dbc.Col(card_component_input("ICE"), md=4),
                    ], )
                ], title="Energy Conversion System Definition I", ),
                #dbc.AccordionItem([], title="Energy Conversion System Definition II"),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_generic_input(component="Financials", header="Financials",
                                                   ident=["discountrate", "lifetime", "operatinghoursyearly"],
                                                   text=["Discount Rate [%]",
                                                         "Lifetime [y]",
                                                         "Operating hours [hr/yr]"]), md=4),
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Fuel Cost Settings"),
                            dbc.CardBody([
                                dbc.Row([dbc.Col(generic_dropdown(id="dd_NH3_fuel_cost", label="NH3 Cost Selector",
                                                                  elements=NH3_fuel_cost_Cases)),
                                         dbc.Col(html.P(id="txt_NH3_fuel_cost_Preset_Selection")),
                                         dbc.Col(generic_dropdown(id="dd_NG_fuel_cost", label="NG Cost Selector",
                                                                  elements=NG_fuel_cost_Cases))]),
                                dbc.Col(html.P(id="txt_NG_fuel_cost_Preset_Selection")),
                                dbc.Row([
                                    dbc.Col(width=6),
                                    dbc.Col(dbc.Label("Nominal"), width=2),
                                    dbc.Col(dbc.Label("Min"), width=2),
                                    dbc.Col(dbc.Label("Max"), width=2)]),

                                input_row1(component="Fuel", ident="NH3_cost_EUR_per_kW", text="NH3 cost [€/kWh]"),
                                input_row1(component="Fuel", ident="NH3_costIncrease_percent_per_year",
                                           text="NH3 cost increase [%/yr]"),
                                input_row1(component="Fuel", ident="NG_cost_EUR_per_kW", text="NG cost [€/kWh]"),
                                input_row1(component="Fuel", ident="NG_costIncrease_percent_per_year",
                                           text="NG cost increase [%/yr]")
                            ])
                        ]), md=4
                        ),

                    ])
                ], title="General Settings", ),
                dbc.AccordionItem([], title="LCOE Plots"),
                dbc.AccordionItem([], title="LCOE Sensitivity Study"),
                dbc.AccordionItem([], title="About"),
                dbc.AccordionItem([dbc.Row([dbc.Col(dbc.Button("Initial Data Collect", id="bt_collect"), width=2),
                                            dbc.Col(dbc.Button("Update Data Collect", id="bt_update_collect"), width=2),
                                            dbc.Col(dbc.Button("Export Input", id="bt_export_Input"), width=2),
                                            dbc.Col(dbc.Button("Load Input", id="bt_load_Input"), width=2)
                                            ]),
                                   dbc.Row([html.Pre("...", id="txt_out1")]), # ToDo
                                   dbc.Row([html.Pre("...", id="txt_out2")]),
                                   dbc.Row([html.Pre("...", id="txt_out3")]),
                                    dbc.Row([html.Pre("...", id="txt_out4")]),
                                    dbc.Row([html.Pre("...", id="txt_out5")])
                                   ], title="Developer"),

            ], always_open=True)
        ]),
    ]),

], fluid=True)


# Developer Callbacks
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------
@app.callback(
    Output("txt_out1", "children"), Input("bt_collect", "n_clicks"), State({'type': 'input', 'index': ALL}, 'value'),
    prevent_initial_call=True)
def dev_button_initialCollectInput(input, states):
    df = pd.DataFrame(index=ctx.states.keys(), columns=["input_name"])

    for key, val in ctx.states.items():
        par = key.split('"')[3]
        df.loc[key, "input_name"] = par
        df.loc[key, 0] = val
    df.to_pickle("data.pkl")
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
    State({'type': 'input', 'index': ALL}, 'value'), prevent_initial_call=True)
def dev_button_exportInput(input, states):
    for key, val in ctx.states.items():
        df_definitions.loc[key, input] = val
    df_definitions.to_excel("test.xlsx")
    return "ok"

@app.callback(
    Output({'type': 'input', 'index': ALL}, 'value'), Input("bt_load_Input", "n_clicks"), prevent_initial_call=True)
def dev_button_loadInput(input):
    # Input field list:
    # Reihenfolge der ids
    #print(ctx.outputs_list[0]["id"]["index"])

    df = pd.read_pickle("data_return.pkl")
    return_list = []
    for el in ctx.outputs_list:
        el_id_index = el["id"]["index"]
        return_list.append(df.loc[df.input_name == el_id_index, 100].item())
    #print(return_list)
    return return_list


@app.callback(
    Output("txt_Preset_Selection", "children"), [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(preset_Cases))])
def show_selected_preset(a, b):
    if ctx.triggered_id is not None:
        selection_id = preset_Cases[int(ctx.triggered_id[-1])]
    else:
        selection_id = 'No Selection'

    return f"{selection_id}"


# @app.callback(
#    Output("txt_NH3_fuel_cost_Preset_Selection", "children"), [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(NH3_fuel_cost_Cases))]
# )
# def show_selected_NH3preset(a, b):
#     if ctx.triggered_id is not None:
#         selection_id = NH3_fuel_cost_Cases[int(ctx.triggered_id[-1])]
#     else:
#         selection_id = 'No Selection'
#
#     return f"{selection_id}"
#
# @app.callback(
#     Output("txt_NG_fuel_cost_Preset_Selection", "children"), [Input(f"dd_NG_fuel_cost{n}", "n_clicks") for n in range(len(NG_fuel_cost_Cases))]
# )
# def show_selected_NGpreset(a, b):
#     if ctx.triggered_id is not None:
#         selection_id = NG_fuel_cost_Cases[int(ctx.triggered_id[-1])]
#     else:
#         selection_id = 'No Selection'
#
#     return f"{selection_id}"


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
