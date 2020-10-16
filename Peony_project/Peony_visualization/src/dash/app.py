import dash
import os
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import datetime
import re

from dash.dependencies import Input, Output, State
from PeonyPackage.PeonyDb import MongoDb
from Peony_database.src.database_results.results_summary import PeonyDbResults

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
api = MongoDb()
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
database_results = PeonyDbResults()


def visualize_auc_evolutions(
    auc_seq_passive_1,
    auc_seq_passive_2,
    auc_seq_active_1,
    auc_seq_active_2,
    model_1,
    model_2,
    title,
):

    auc_1_passive_mean = np.mean(auc_seq_passive_1, axis=0).reshape(-1)
    auc_1_passive_std = np.std(auc_seq_passive_1, axis=0).reshape(-1)

    auc_2_passive_mean = np.mean(auc_seq_passive_2, axis=0).reshape(-1)
    auc_2_passive_std = np.std(auc_seq_passive_2, axis=0).reshape(-1)

    auc_1_active_mean = np.mean(auc_seq_active_1, axis=0).reshape(-1)
    auc_1_active_std = np.std(auc_seq_active_1, axis=0).reshape(-1)

    auc_2_active_mean = np.mean(auc_seq_active_2, axis=0).reshape(-1)
    auc_2_active_std = np.std(auc_seq_active_2, axis=0).reshape(-1)

    fig = go.Figure()
    fig.layout.template = "plotly_dark"

    # Passive learning

    y_upper = auc_1_passive_mean + auc_1_passive_std
    y_lower = auc_1_passive_mean - auc_1_passive_std
    fig.add_trace(
        go.Scatter(
            x=list(range(200)) + list(range(200))[::-1],
            y=np.concatenate(
                [y_upper, y_lower[::-1]],
            ),
            line=dict(color="rgb(171, 235, 198)", dash="dash", width=1),
            fill="toself",
            fillcolor="rgba(234, 250, 241, 0.2)",
            name=f"Random Selection {model_1} mean",
            showlegend=False,
        )
    )

    y_upper = auc_2_passive_mean + auc_2_passive_std
    y_lower = auc_2_passive_mean - auc_2_passive_std
    fig.add_trace(
        go.Scatter(
            x=list(range(200)) + list(range(200))[::-1],
            y=np.concatenate(
                [y_upper, y_lower[::-1]],
            ),
            line=dict(color="rgb(210, 180, 222)", dash="dash", width=1),
            fill="toself",
            fillcolor="rgba(244, 236, 247, 0.2)",
            name=f"Random Selection {model_2} mean",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(200)),
            y=auc_1_passive_mean,
            name=f"Random Selection {model_1} mean",
            line=dict(
                color="rgb(46, 204, 113)",
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(200)),
            y=auc_2_passive_mean,
            name=f"Random Selection {model_2} mean",
            line=dict(
                color="rgb(165, 105, 189)",
                width=2,
            ),
        )
    )

    # Active learning

    y_upper = auc_1_active_mean + auc_1_active_std
    y_lower = auc_1_active_mean - auc_1_active_std
    fig.add_trace(
        go.Scatter(
            x=list(range(200)) + list(range(200))[::-1],
            y=np.concatenate(
                [y_upper, y_lower[::-1]],
            ),
            line=dict(color="rgb(130, 224, 170)", dash="dash", width=1),
            fill="toself",
            fillcolor="rgba(213, 245, 227, 0.5)",
            name=f"Random Selection {model_1} mean",
            showlegend=False,
        )
    )

    y_upper = auc_2_active_mean + auc_2_active_std
    y_lower = auc_2_active_mean - auc_2_active_std
    fig.add_trace(
        go.Scatter(
            x=list(range(200)) + list(range(200))[::-1],
            y=np.concatenate(
                [y_upper, y_lower[::-1]],
            ),
            line=dict(color="rgb(187, 143, 206)", dash="dash", width=1),
            fill="toself",
            fillcolor="rgba(232, 218, 239, 0.5)",
            name=f"Random Selection {model_2} mean",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(200)),
            y=auc_1_active_mean,
            name=f"Active Learning {model_1} mean",
            line=dict(
                color="rgb(35, 155, 86)",
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(200)),
            y=auc_2_active_mean,
            name=f"Active Learning {model_2} mean",
            line=dict(
                color="rgb(108, 52, 131)",
                width=2,
            ),
        )
    )

    fig.update_layout(
        title_text=f"{title} categories",
        xaxis_title="AUC",
        yaxis_title="Learning Iterations",
    )

    fig.update_layout(
        height=500, width=1000, legend=dict(y=0.01, xanchor="right", x=0.99)
    )

    return fig


def create_plot(categories_list, alg_1, alg_2):

    list_of_plots = []

    # alg_1 = "bayesian_dropout_nn_fast_text_embeddings"
    # alg_2 = "bayesian_denfi_v_2_0.3_fast_text_embeddings"

    alg_legend_1 = " ".join([token.capitalize() for token in alg_1.split("_")[:2]])
    alg_legend_2 = " ".join([token.capitalize() for token in alg_2.split("_")[:2]])

    for index, categories_string in enumerate(categories_list):

        category_1, category_2 = [
            val.strip().upper()
            for val in re.sub("_", " ", categories_string).split("vs")
        ]
        title_category = " ".join(
            [category_1.capitalize(), "Vs.", category_2.capitalize()]
        )

        # Random acquisition function
        random_sampling_results_1 = api.get_model_results(
            {
                "model": alg_1,
                "acquisition_function": "random",
                "category_1": category_1,
                "category_2": category_2,
            }
        )[0]
        random_sampling_results_2 = api.get_model_results(
            {
                "model": alg_2,
                "acquisition_function": "random",
                "category_1": category_1,
                "category_2": category_2,
            }
        )[0]

        # Entropy acquisition function
        entropy_sampling_results_1 = api.get_model_results(
            {
                "model": alg_1,
                "acquisition_function": "entropy",
                "category_1": category_1,
                "category_2": category_2,
            }
        )[0]
        entropy_sampling_results_2 = api.get_model_results(
            {
                "model": alg_2,
                "acquisition_function": "entropy",
                "category_1": category_1,
                "category_2": category_2,
            }
        )[0]

        list_of_plots.append(
            dcc.Graph(
                id=f"graph_{index}",
                figure=visualize_auc_evolutions(
                    random_sampling_results_1["results"],
                    random_sampling_results_2["results"],
                    entropy_sampling_results_1["results"],
                    entropy_sampling_results_2["results"],
                    alg_legend_1,
                    alg_legend_2,
                    title_category,
                ),
            )
        )

    return list_of_plots


app.layout = html.Div(
    className="row",
    children=[
        dcc.Store(id="visualized-categories", storage_type="session"),
        dcc.Store(id="tmp", storage_type="session"),
        html.Div(
            className="div-left-panel",
            children=[
                # HEADLINE
                html.H5(
                    children="Peony Visualization Component",
                    style={"text-align": "center"},
                ),
                # LOGO AND DESCRIPTION
                html.Div(
                    className="div-info",
                    children=[
                        html.Img(
                            className="logo",
                            src=app.get_asset_url("peony-logo.png"),
                        ),
                        html.P(
                            """
                            This tool is made to visualize and compare different 
                            evolutions of the active learning algorythms. 
                            """,
                            style={"text-align": "center"},
                        ),
                    ],
                ),
                # DROPDOWN
                html.Div(
                    className="drop-downs",
                    children=[
                        html.P(
                            children="Available Categories",
                            style={"text-align": "center"},
                        ),
                        dcc.Dropdown(
                            id="categories-dropdown",
                            options=[
                                {
                                    "label": "Crime Vs. Good News",
                                    "value": "crime_vs_good_news",
                                },
                                {
                                    "label": "Sports Vs. Comedy",
                                    "value": "sports_vs_comedy",
                                },
                                {
                                    "label": "Politics Vs. Business",
                                    "value": "politics_vs_business",
                                },
                                {
                                    "label": "Tech Vs. Science",
                                    "value": "tech_vs_science",
                                },
                                {
                                    "label": "College Vs. Education",
                                    "value": "college_vs_education",
                                },
                                {
                                    "label": "Positive Vs. Negative Tweets",
                                    "value": "positive_emotions_tweets_vs_negative_emotions_tweets",
                                },
                                {
                                    "label": "Reset",
                                    "value": "reset",
                                },
                            ],
                            # value="crime_vs_good_news",
                            clearable=False,
                            style={
                                "width": "200pt",
                            },
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.P(
                            children="MongoDb Stored Algorithms",
                            style={"text-align": "center"},
                        ),
                        dcc.Dropdown(
                            id="first-alg-dropdown",
                            options=[
                                {
                                    "label": " ".join(
                                        [token.capitalize() for token in key.split("_")]
                                    ),
                                    "value": key,
                                }
                                for key in sorted(database_results.structurize_data())
                            ],
                            optionHeight=70,
                            clearable=True,
                            style={
                                "width": "100%",
                                "display": "inline-block",
                                "font-size": "100%",
                            },
                        ),
                        dcc.Dropdown(
                            id="second-alg-dropdown",
                            options=[
                                {
                                    "label": " ".join(
                                        [token.capitalize() for token in key.split("_")]
                                    ),
                                    "value": key,
                                }
                                for key in sorted(database_results.structurize_data())
                            ],
                            optionHeight=70,
                            style={
                                "width": "100%",
                                "display": "inline-block",
                                "font-size": "100%",
                                "align-items": "center",
                                # "height": "50pt",
                            },
                        ),
                    ],
                    style={
                        "width": "200pt",
                        "display": "inline-block",
                        "margin-top": "20pt",
                    },
                ),
                # CLOCK
                # html.P(
                #     id="live_clock",
                #     children=str(datetime.datetime.now().strftime("%H:%M:%S")),
                #     style={
                #         "margin-top": "100pt",
                #     },
                # ),
            ],
            style={
                "width": "200pt",
                "display": "inline-block",
                "height": "2000pt",
            },  # "position": "fixed"},
        ),
        # RIGHT PANEL WITH CHARTS
        html.Div(
            id="charts",
            className="learning-chart",
            style={
                "display": "inline-block",
                "horizontal-align": "right",
                "margin-left": "1%",
            },
            children=[
                html.Div(
                    id="div-w-plots",
                )
            ],
        ),
    ],
    style={"display": "flex"},
)


@app.callback(
    Output("tmp", "data"),
    [
        Input("visualized-categories", "data"),
    ],
)
def create_temp_variable(categories):
    if categories is None:  # isinstance(categories, dict):
        return ["crime_vs_good_news"]
    return categories


@app.callback(
    [Output("div-w-plots", "children"), Output("visualized-categories", "data")],
    [
        Input("categories-dropdown", "value"),
        Input("first-alg-dropdown", "value"),
        Input("second-alg-dropdown", "value"),
        Input("tmp", "data"),
    ],
)
def update_figure(categories_string, first_alg, second_alg, categories):
    if categories_string is None or categories_string == "reset":
        return [], []
    if first_alg is None or second_alg is None:
        return [], []
    if categories_string not in categories:
        categories.append(categories_string)
    return create_plot(categories, first_alg, second_alg), categories


if __name__ == "__main__":
    app.run_server(host=os.getenv("localhost", "127.0.0.1"), debug=True)
