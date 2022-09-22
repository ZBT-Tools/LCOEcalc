"""
Dash port of Shiny iris k-means example:

https://shiny.rstudio.com/gallery/kmeans-example.html
"""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html, ctx

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

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def input_row1(component, property):
    row = dbc.Row([
        dbc.Col(dbc.Label(property), width=6),
        dbc.Col(dbc.Input(id=f"input_{component}_{property}", type="text", size="sm"), width=2),
        dbc.Col(dbc.Input(id=f"input_{component}_{property}_min", type="text", disabled=True, size="sm"), width=2),
        dbc.Col(dbc.Input(id=f"input_{component}_{property}_max", type="text", disabled=True, size="sm"), width=2)])
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
                input_row1(component=name, property="Capex [€/kW]"),
                input_row1(component=name, property="Opex (no Fuel) [€/kW]"),
                input_row1(component=name, property="Efficiency [%]"),
            ])])])
    return card


def card_generic_input(component: str, header: str, properties: list):
    # Create Input rows
    rows = [dbc.Col(width=6),
            dbc.Col(dbc.Label("Nominal"), width=2)]
    rows.extend([input_row1(component=component, property=a) for a in properties])

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
                ], title="Energy Conversion System Definition", ),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_generic_input(component="Financials", header="Financials",
                                                   properties=["Discount Rate [%]",
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

                                input_row1(component="Fuel", property="NH3 cost [€/kWh]"),
                                input_row1(component="Fuel", property="NH3 cost increase [%/yr]"),
                                input_row1(component="Fuel", property="NG cost [€/kWh]"),
                                input_row1(component="Fuel", property="NG cost increase [%/yr]")
                            ])
                        ])
                        ),

                        dbc.Col(card_generic_input(component="Fuel", header="Fuel Definitions",
                                                   properties=["Cost [€/kWh]",
                                                               "Yearly increase [%]",
                                                               ]), md=4),

                    ])
                ], title="General Settings", ),
                dbc.AccordionItem([dbc.Row([dbc.Col(dbc.Button("Primary", id="bt1_update"), width=2),
                                            dbc.Col(dbc.Button("Secondary", id="bt2_update"), width=2)]),
                                   dbc.Row([html.P("...", id="txt_out")]),
                                   ], title="LCOE Plots"),
                dbc.AccordionItem([], title="LCOE Sensitivity Study"),
                dbc.AccordionItem([], title="About"),

            ], always_open=True)
        ]),
    ]),

], fluid=True)


@app.callback(
    Output("txt_out", "children"), [Input("bt1_update", "n_clicks"), Input("bt2_update", "n_clicks")]
)
def on_button_click(n, m):
    if (n is None) or (m is None):
        return "Not clicked."
    else:
        return f"Clicked {n} and {m} times."


@app.callback(
    Output("txt_Preset_Selection", "children"), [Input(f"dd_preset_{n}", "n_clicks") for n in range(len(preset_Cases))]
)
def show_selected_preset(a, b):
    if ctx.triggered_id is not None:
        selection_id = preset_Cases[int(ctx.triggered_id[-1])]
    else:
        selection_id = 'No Selection'

    return f"{selection_id}"

#@app.callback(
#    Output("txt_NH3_fuel_cost_Preset_Selection", "children"), [Input(f"dd_NH3_fuel_cost_{n}", "n_clicks") for n in range(len(NH3_fuel_cost_Cases))]
#)
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
