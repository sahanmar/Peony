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
from plotly.colors import n_colors
from PeonyPackage.PeonyDb import MongoDb
from Peony_database.src.database_results.results_summary import PeonyDbResults

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
api = MongoDb()
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
database_results = PeonyDbResults()

tabs_styles = {"height": "44px"}

tab_style = {
    "borderBottom": "1px solid #d6d6d6",
    "padding": "6px",
}

tab_selected_style = {
    "borderTop": "1px solid #d6d6d6",
    "borderBottom": "1px solid #d6d6d6",
    "backgroundColor": "#1f2e2e",
    "color": "white",
    "padding": "6px",
}


def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


def find_similar_algorithms_in_db(alg):
    db_algs = database_results.structurize_data()
    similar_algs = []
    for db_alg in db_algs:
        str_non_similarity = hamming_distance(alg, db_alg)
        if str_non_similarity <= 2:
            similar_algs.append(db_alg)
    return sorted(similar_algs)


def create_evolution_table_stats(res, mean_func, dev_func, slider_val):
    slider_val = slider_val - 1
    return [round(mean_func(res, axis=0)[0][0], 3)] + [
        round(mean_func(res, axis=0)[int((i * 200 / (slider_val) - 1))][0], 3)
        for i in range(1, slider_val + 1)
    ]


def get_res_from_db(alg, acsq_func, category_1, category_2):
    res = api.get_model_results(
        {
            "model": alg,
            "acquisition_function": acsq_func,
            "category_1": category_1,
            "category_2": category_2,
        }
    )
    return res[0]["results"] if res else None


def create_evloution_table(category, algs, slider_val):

    dev_func = np.std
    mean_func = np.mean

    alg_legend = " ".join([token.capitalize() for token in algs[0].split("_")[:2]])
    category_1, category_2 = [
        val.strip().upper() for val in re.sub("_", " ", category).split("vs")
    ]

    title_category = " ".join([category_1.capitalize(), "Vs.", category_2.capitalize()])

    if len(algs) == 1:
        return html.H4(
            f"No additional noise visualization data in Db found for {title_category} and {alg_legend}"
        )

    list_w_results = []

    for alg in algs:
        res = get_res_from_db(alg, "entropy", category_1, category_2)
        if res is None:
            return html.H4(
                f"No additional noise visualization data in Db found for {title_category} and {alg_legend}"
            )
        list_w_results.append(
            create_evolution_table_stats(res, mean_func, dev_func, slider_val)
        )

    list_w_results = list(map(list, zip(*list_w_results)))
    noise_var = [0.1, 0.2, 0.3, 0.4, 0.6]
    colors = n_colors("rgb(172, 193, 198)", "rgb(2, 52, 81)", 5, colortype="rgb")
    table_to_vis = [noise_var] + [val for val in list_w_results]
    slider_val -= 1
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Noise Variance</b>", "<b>0</b>"]
                    + [
                        f"<b>{int(i * 200 / (slider_val))}</b>"
                        for i in range(1, slider_val + 1)
                    ],
                    font_size=13,
                    align="center",
                    height=40,
                ),
                cells=dict(
                    values=table_to_vis,
                    fill_color=[
                        np.array(colors)[np.argsort(np.argsort(val))]
                        for val in table_to_vis
                    ],
                    align="center",
                    font_size=13,
                    height=40,
                ),
            )
        ]
    )

    fig.update_layout(
        height=500,
        width=1000,
        title_text=f"{alg_legend} Algorithm with Respect to Different Additive Noise",
    )
    fig.layout.template = "plotly_dark"
    return dcc.Graph(figure=fig)


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
        yaxis_title="AUC",
        xaxis_title="Learning Iterations",
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
        random_sampling_results_1 = get_res_from_db(
            alg_1, "random", category_1, category_2
        )
        random_sampling_results_2 = get_res_from_db(
            alg_2, "random", category_1, category_2
        )

        # Entropy acquisition function
        entropy_sampling_results_1 = get_res_from_db(
            alg_1, "entropy", category_1, category_2
        )
        entropy_sampling_results_2 = get_res_from_db(
            alg_2, "entropy", category_1, category_2
        )

        list_of_plots.append(
            dcc.Graph(
                id=f"graph_{index}",
                figure=visualize_auc_evolutions(
                    random_sampling_results_1,
                    random_sampling_results_2,
                    entropy_sampling_results_1,
                    entropy_sampling_results_2,
                    alg_legend_1,
                    alg_legend_2,
                    title_category,
                ),
            )
            if all(
                [
                    random_sampling_results_1,
                    random_sampling_results_2,
                    entropy_sampling_results_1,
                    entropy_sampling_results_2,
                ]
            )
            else html.H5(f"No data in Db found for {title_category} AUC visualization")
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
                "width": "100%",
            },
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    children=[
                        dcc.Tab(
                            label="AUC Evolutions",
                            value="tab-1",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                html.Div(
                                    id="div-w-plots",
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="Additive Noise Results Fluctuation",
                            value="tab-2",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                html.Div(
                                    children=[
                                        html.H5(
                                            children=[
                                                "Slide bar to visualize number of table columns"
                                            ],
                                            style={
                                                "text-align": "center",
                                                "margin-bottom": "10pt",
                                                "margin-top": "10pt",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="slider",
                                            min=2,
                                            max=10,
                                            value=5,
                                            marks={
                                                str(i): str(i) for i in range(2, 11)
                                            },
                                            step=None,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    style={"display": "flex", "width": "100%"},
                    colors={
                        "background": "black",
                    },
                ),
                html.Div(id="tabs-content-classes"),
            ],
        ),
    ],
    style={"display": "flex"},
)


def intro_page():
    return [
        html.Div(
            children=[
                html.H6(
                    children=[
                        """
                        Peony Visualization Component serves as an endpoint that queries MongoDb and visualizes maching learning results.
                        The visualizatopn component is implemented in Dash by PlotLy.
                        """
                    ],
                    style={
                        # "text-align": "center",
                        "margin-bottom": "20pt",
                        "margin-top": "40pt",
                        "margin-right": "10%",
                        "margin-left": "10%",
                    },
                ),
                html.Div(
                    children=[
                        html.Img(
                            className="dash-mongo-logo",
                            src=app.get_asset_url("dash-mongo.png"),
                            style={
                                "height": "100%",
                                "width": "100%",
                                # "display": "inline-block",
                            },
                        ),
                    ],
                ),
                html.H6(
                    children=[
                        """
                        The tool allows a user to visualize AUC (Area Under ROC curve) evolutions and Additive Noise Fluctuations with respect to different categories.
                        
                        Categories and results for different algorithms can be found in dropdown menus. 

                        Both plots and tables are interactive. Thus, a user is able to extract more information. Moreover, a user can also download a plot or a table.
                        """
                    ],
                    style={
                        # "text-align": "center",
                        "margin-bottom": "20pt",
                        "margin-top": "20pt",
                        "margin-right": "10%",
                        "margin-left": "10%",
                    },
                ),
                html.H6(
                    children=["""Author: Marko Sahan"""],
                    style={
                        "margin-top": "40pt",
                        "margin-right": "10%",
                        "margin-left": "10%",
                    },
                ),
                html.A(
                    "Peony Project GitHub",
                    href="https://github.com/sahanmar/Peony/",
                    style={
                        "margin-right": "10%",
                        "margin-left": "10%",
                    },
                ),
            ]
        )
    ]


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
    [
        Output("tabs-content-classes", "children"),
        Output("visualized-categories", "data"),
    ],
    [
        Input("categories-dropdown", "value"),
        Input("first-alg-dropdown", "value"),
        Input("second-alg-dropdown", "value"),
        Input("tabs", "value"),
        Input("slider", "value"),
        Input("tmp", "data"),
    ],
)
def update_figure(
    categories_string, first_alg, second_alg, tab, slider_val, categories
):
    if tab == "tab-1":
        if categories_string is None or categories_string == "reset":
            return intro_page(), []
        if first_alg is None or second_alg is None:
            return intro_page(), []
        if categories_string not in categories:
            categories.append(categories_string)
        return create_plot(categories, first_alg, second_alg), categories
    else:
        if categories is None or first_alg is None or second_alg is None:
            return intro_page(), categories
        else:
            if categories_string is None or categories_string == "reset":
                return intro_page(), []
            if categories_string not in categories:
                categories.append(categories_string)
            alg_1_similar = find_similar_algorithms_in_db(first_alg)
            alg_2_similar = find_similar_algorithms_in_db(second_alg)
            return (
                [
                    html.Div(
                        id=f"table_{index}",
                        children=[
                            create_evloution_table(category, alg_1_similar, slider_val),
                            create_evloution_table(category, alg_2_similar, slider_val),
                        ],
                    )
                    for index, category in enumerate(categories)
                ],
                categories,
            )


if __name__ == "__main__":
    app.run_server(
        host=os.getenv("localhost", "127.0.0.1"),
        debug=True,
        dev_tools_ui=False,
    )
