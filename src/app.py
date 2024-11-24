'''
Filipe Chagas Ferraz (github.com/FilipeChagasDev)
Nov-2024
'''
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from ant_colony import AntColonyOptimizer
from ui_components import numeric_input, input_group, upload_group
from ui_components import pointset_plot, solution_plot, solutions_over_time_plot, length_over_time_plot
from ui_components import pheromone_distribution_plot, entropy_over_time_plot
from data_functions import euclidean_distance_matrix, indices_to_names
from help import help_article


app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
app.title = 'SAMSA'
server = app.server

# --- DATA ---

CRITICAL_N_POINTS = 10
DEFAULT_N_POINTS = 10
DEFAULT_MIN_X = -1
DEFAULT_MAX_X = 1
DEFAULT_MIN_Y = -1
DEFAULT_MAX_Y = 1

data = {
    'points_df': pd.DataFrame({
            'name': [f'P{i}' for i in range(DEFAULT_N_POINTS)],
            'x': np.random.random(DEFAULT_N_POINTS)*(DEFAULT_MAX_X-DEFAULT_MIN_X) + DEFAULT_MIN_X,
            'y': np.random.random(DEFAULT_N_POINTS)*(DEFAULT_MAX_Y-DEFAULT_MIN_Y) + DEFAULT_MIN_Y,
        })
}


# --- SIDEBAR LAYOUT ---

SIDEBAR_WIDTH = '25rem'

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": SIDEBAR_WIDTH,
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    'overflow-y': 'scroll'
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": SIDEBAR_WIDTH,
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        dbc.Row(children=[
            dbc.Col(html.H4("𖢥 SAMSA"), width='auto'),
            dbc.Col(html.A("Developed by Filipe Chagas", href='https://filipechagasdev.github.io/FilipeChagasDev/'))
        ], align='center'),
        html.Hr(),
        dbc.Nav(
            children=[
                dbc.Accordion(className='mb-4', children=[
                    dbc.AccordionItem(title='📍 Pointset', children=[
                        #upload_group(),
                        input_group('📝 Random pointset generation', [
                            [
                                numeric_input('Min X', value=-1, min=-100, max=100, step=1, id='min_x'),
                                numeric_input('Max X', value=1, min=-100, max=100, step=1, id='max_x')
                            ],
                            [
                                numeric_input('Min Y', value=-1, min=-100, max=100, step=1, id='min_y'),
                                numeric_input('Max Y', value=1, min=-100, max=100, step=1, id='max_y')
                            ],
                            [numeric_input('Number of points', value=10, min=4, max=1000, step=1, id='n_points')],
                            [dbc.Button('Generate', className='mb-4', id='generate-btn')]
                        ]),
                    ]),
                    dbc.AccordionItem(title='🐜 Ant Colony Optimization parameters', children=[
                        input_group('⏳ Time and resources', [[
                            numeric_input('Number of ants', value=20, min=1, max=10000, step=1, id='n_ants'),
                            numeric_input('Number of epochs', value=50, min=1, max=10000, step=1, id='n_epochs')
                        ]]),
                        input_group('🎲 Probability', [[
                            numeric_input('α (alpha)', value=1, min=0.1, max=10, step=0.1, id='alpha'),
                            numeric_input('β (beta)', value=2, min=0.1, max=10, step=0.1, id='beta')
                        ]]),
                        input_group('👣 Pheromone', [[
                            numeric_input('ρ (rho)', value=0.10, min=0.01, max=0.99, step=0.01, id='rho'),
                            numeric_input('ζ (zeta)', value=-2, min=-5, max=5, step=0.25, id='zeta')
                        ]]),
                    ])
                ]),
                dbc.Button("Solve TSP Instance", color="primary", id='solve-btn')
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# --- PROBLEM INSTANCE TAB ---

problem_instance_tab = dbc.Tab(
    label="Problem Instance", 
    tab_id="instance", 
    children=[
        dbc.Row(children=[
            dbc.Col(dcc.Graph(
                id='pointset_plot',
                figure=pointset_plot(data['points_df']),
                style={
                    'width': '100%',  # Largura do gráfico ocupa 100% do espaço disponível
                    'height': '90vh',  # Altura do gráfico ocupa 90% da altura da janela de visualização
                }
            )),
        ])                    
    ]
)



# --- SOLUTION TAB ---

solution_tab = dbc.Tab(label="Solution", tab_id="solution", id='solution_tab', children=[
    dcc.Graph(
        id='solution_plot',
        figure=go.Figure(),
        style={
            'width': '100%',  # Largura do gráfico ocupa 100% do espaço disponível
            'height': '90vh',  # Altura do gráfico ocupa 90% da altura da janela de visualização
        }
    ),
    dbc.Card(children='', id='solution_sequence', className='m-4 p-4')
])

# --- OPTIMIZER STATES OVER TIME TAB ---

solution_over_time_tab = dbc.Tab(label="Solution over time",
    tab_id="solution_over_time", 
    id='solution_over_time', 
    children=[
        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Graph(
                    id='solution_over_time_plot',
                    figure=go.Figure(),
                    style={
                        'width': '100%',
                        'height': '90vh'
                    }
                )
            ]),
        ]),
        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Graph(
                    id='length_over_time_plot',
                    figure=go.Figure(),
                    style={
                        'width': '100%',
                        'height': '90vh'
                    }
                )
            ]),
        ])
    ]
)


# --- OPTIMIZER STATES OVER TIME TAB ---

pheromone_over_time_tab = dbc.Tab(label="Pheromone distribution over time",
    tab_id="pheromone_over_time", 
    id='pheromone_over_time', 
    children=[
        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Graph(
                    id='pheromone_over_time_plot',
                    figure=go.Figure(),
                    style={
                        'width': '100%',
                        'height': '90vh'
                    }
                )
            ]),
        ]),
        dbc.Row(children=[
            dbc.Col(children=[
                dcc.Graph(
                    id='entropy_over_time_plot',
                    figure=go.Figure(),
                    style={
                        'width': '100%',
                        'height': '90vh'
                    }
                )
            ]),
        ])
    ]
)


# --- HELP TAB ---

help_tab = dbc.Tab(label="Help",
    tab_id="help_tab", 
    id='help_tab', 
    children=[
        dcc.Markdown(help_article, mathjax=True, className='m-4')
    ]
)

# --- PAGE LAYOUT ---

tabs = dbc.Tabs(children=[
        problem_instance_tab, 
        solution_tab,
        solution_over_time_tab,
        pheromone_over_time_tab,
        help_tab
    ],
    id="tabs",
    active_tab="instance",
)

content = html.Div(children=[
        dcc.Loading(overlay_style={"visibility": "visible", "filter": "blur(2px)"}, children=[
            tabs
        ])
    ], 
    id="page-content", 
    style=CONTENT_STYLE
)


app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# --- CALLBACKS ---

# Callback to change the tab to "solution" when "Solve TSP Instance" is clicked
@app.callback(
    Output('solution_plot', 'figure'),
    Output('solution_over_time_plot', 'figure'),
    Output('length_over_time_plot', 'figure'),
    Output('pheromone_over_time_plot', 'figure'),
    Output('entropy_over_time_plot', 'figure'),
    Output('solution_sequence', 'children'),
    Output('tabs', 'active_tab', allow_duplicate=True),
    Input('solve-btn', 'n_clicks'),
    State('n_ants', 'value'),
    State('n_epochs', 'value'),
    State('alpha', 'value'),
    State('beta', 'value'),
    State('rho', 'value'),
    State('zeta', 'value'),
    prevent_initial_call=True
)
def solve_instance_callback(n_clicks, n_ants, n_epochs, alpha, beta, rho, zeta):
    if n_clicks:
        distance_matrix = euclidean_distance_matrix(data['points_df'])
        aco = AntColonyOptimizer(distance_matrix, n_ants, n_epochs, alpha, beta, rho, zeta)
        path_hist = []
        best_len_hist = []
        phero_hist = []
        path, length = aco.optimize(path_history=path_hist, best_length_history=best_len_hist, pheromone_history=phero_hist)

        out1 = solution_plot(data['points_df'], path)
        out2 = solutions_over_time_plot(data['points_df'], np.array(path_hist))
        out3 = length_over_time_plot(best_len_hist)
        if len(data['points_df']) <= CRITICAL_N_POINTS:
            out4 = pheromone_distribution_plot(data['points_df'], phero_hist)
        else:
            fig = go.Figure()
            fig.add_annotation(text=f"This chart can only be displayed for up to {CRITICAL_N_POINTS} points.", x=0.5, y=0.5, showarrow=False, font=dict(size=24), align="center")
            fig.update_layout(xaxis=dict(range=[0, 1], showgrid=False), yaxis=dict(range=[0, 1], showgrid=False), plot_bgcolor="white")
            out4=fig
        out5 = entropy_over_time_plot(phero_hist)
        out6 = html.Div(children=[
            html.P(children=[
                html.Span('Path: ', style={'font-weight': 'bold'}), 
                ', '.join(indices_to_names(path, data['points_df']))
            ]),
            html.P(children=[
                html.Span('Length: ', style={'font-weight': 'bold'}), 
                str(length)
            ])
        ])
        out7 = 'solution'
        return out1, out2, out3, out4, out5, out6, out7
    
    return go.Figure(), go.Figure(), go.Figure(), '', 'instance'


@app.callback(
        Output('pointset_plot', 'figure'),
        Output('tabs', 'active_tab', allow_duplicate=True),
        Input('generate-btn', 'n_clicks'), 
        State('n_points', 'value'),
        State('min_x', 'value'),
        State('max_x', 'value'),
        State('min_y', 'value'),
        State('max_y', 'value'),
        prevent_initial_call=True
)
def generate_pointset(n_clicks, n_points, min_x, max_x, min_y, max_y):
    if n_clicks:
        assert max_x > min_x
        assert max_y > min_y

        points_df = pd.DataFrame({
            'name': [f'P{i}' for i in range(n_points)],
            'x': np.random.random(n_points)*(max_x-min_x) + min_x,
            'y': np.random.random(n_points)*(max_y-min_y) + min_y,
        })

        data['points_df'] = points_df

    return pointset_plot(data['points_df']), 'instance'


if __name__ == '__main__':
    app.run(debug=True)