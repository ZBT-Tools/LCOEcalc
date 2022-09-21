"""
Dash port of Shiny iris k-means example:

https://shiny.rstudio.com/gallery/kmeans-example.html
"""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from sklearn import datasets
from sklearn.cluster import KMeans

iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def input_row1(component, property):
    row = dbc.Row([
        dbc.Col(dbc.Label(property), width=6),
        dbc.Col(dbc.Input(id=f"{component}_{property}", type="text", size="sm"), width=2),
        dbc.Col(dbc.Input(id=f"{component}_{property}_min", type="text", disabled=True, size="sm"), width=2),
        dbc.Col(dbc.Input(id=f"{component}_{property}_max", type="text", disabled=True, size="sm"), width=2)])
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


def card_generic_input(name: str, header: str, properties:list ):

    # Create Input rows
    rows = [dbc.Col(width=6),
            dbc.Col(dbc.Label("Nominal"), width=2)]
    rows.extend([input_row1(component=...,property=...) for a in properties])



    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody([
            html.Div([
                dbc.Row(
                    rows
                    [
                    dbc.Col(width=6),
                    dbc.Col(dbc.Label("Nominal"), width=2),

                input_row1(component=name, property="Capex [€/kW]"),
                input_row1(component=name, property="Opex (no Fuel) [€/kW]"),
                input_row1(component=name, property="Efficiency [%]"),
            ])])])
    return card


app.layout = dbc.Container([
    html.H1("HiPowAR LCOE Tool"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                # Energy Conversion Settings
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_component_input("HiPowAR"), md=4),
                        dbc.Col(card_component_input("SOFC"), md=4),
                        dbc.Col(card_component_input("ICE"), md=4),
                    ], )
                ], title="Energy Conversion System Definition", ),
                # General Settings
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(card_component_input("Financial"), md=4),
                        dbc.Col(card_component_input("Fuel"), md=4),
                        dbc.Col(card_component_input("Sonstiges"), md=4),
                    ], )
                ], title="General Settings", )
            ], always_open=True)
        ]),
    ]),
    #  align="center",]),
    dbc.Row(
        [

            # dbc.Col(controls, md=4),
            # dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
        ],
        align="center",
    ),
], fluid=True)

# @app.callback(
#     Output("cluster-graph", "figure"),
#     [
#         Input("ax-variable", "value"),
#         Input("ay-variable", "value"),
#         Input("cluster-count", "value"),
#     ],
# )
# def make_graph(x, y, n_clusters):
#     # minimal input validation, make sure there's at least one cluster
#     km = KMeans(n_clusters=max(n_clusters, 1))
#     df = iris.loc[:, [x, y]]
#     km.fit(df.values)
#     df["cluster"] = km.labels_
#
#     centers = km.cluster_centers_
#
#     data = [
#         go.Scatter(
#             x=df.loc[df.cluster == c, x],
#             y=df.loc[df.cluster == c, y],
#             mode="markers",
#             marker={"size": 8},
#             name="Cluster {}".format(c),
#         )
#         for c in range(n_clusters)
#     ]
#
#     data.append(
#         go.Scatter(
#             x=centers[:, 0],
#             y=centers[:, 1],
#             mode="markers",
#             marker={"color": "#000", "size": 12, "symbol": "diamond"},
#             name="Cluster centers",
#         )
#     )
#
#     layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}
#
#     return go.Figure(data=data, layout=layout)
#
#
# # make sure that x and y values can't be the same variable
# def filter_options(v):
#     """Disable option v"""
#     return [
#         {"label": col, "value": col, "disabled": col == v}
#         for col in iris.columns
#     ]
#
#
# # functionality is the same for both dropdowns, so we reuse filter_options
# app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
#     filter_options
# )
# app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
#     filter_options
# )

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
