'''
Filipe Chagas Ferraz (github.com/FilipeChagasDev)
Nov-2024
'''
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import entropy
import pandas as pd
import numpy as np


# --- SIDEBAR COMPONENTS ---

def numeric_input(label: str, value: float, min: float, max: float, step: float, id: str = None, margin=True):
    return html.Div(children=[
                        label, 
                        dbc.Input(type="number", value=value, min=min, max=max, step=step, id=id)
                    ], className=('mb-4' if margin else None))


def input_group(label: str, grid: list):
    html_grid = [dbc.Row(children=[dbc.Col(col) for col in row]) for row in grid]
    return dbc.Card(children=[html.P(label, className='text-muted')] + html_grid, className='ps-3 pe-3 pt-3 mb-3')


def upload_group():
    upload_comp = dcc.Upload(
        id='upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Excel File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-right': '10px'
        }
    )
    group = input_group('📤 Pointset upload', [[upload_comp], [html.P(html.A('click here to download an example CSV', href='https://github.com/FilipeChagasDev/samsa/releases/download/example.csv/example.csv'), id='upload_fn')]])
    return group


# --- GRAPH COMPONENTS ---

def pointset_plot(points_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=points_df['x'], 
        y=points_df['y'], 
        text=points_df['name'], 
        mode='markers+text',
        textposition='top center',
        marker=dict(size=10, color='#1f77b4', symbol='circle')
    ))
    fig.update_layout(
        title='Traveling salesman stop points',
        title_x=0.5,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        template='plotly_white'
    )
    return fig


def solution_plot(points_df: pd.DataFrame, path_indices: list):
    """
    Plots the points as a scatter plot and draws a closed path connecting the points as per the given indices.

    Parameters:
        points_df (pd.DataFrame): DataFrame containing point names and their coordinates ('x' and 'y').
        path_indices (list): List of indices describing the order of points to be connected in the path.
    """
    # Extract the coordinates based on the provided path indices
    path_coords = points_df[['x', 'y']].iloc[path_indices].values

    # Extract the x and y coordinates for the path and points
    x_path = path_coords[:, 0]
    y_path = path_coords[:, 1]
    
    # Add the first point to the end to close the path
    x_path = list(x_path) + [x_path[0]]
    y_path = list(y_path) + [y_path[0]]
    
    # Create the scatter plot for the points
    fig = go.Figure()

    # Add the path as a line connecting the points in the order specified
    fig.add_trace(go.Scatter(
        x=x_path,
        y=y_path,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
    ))

    # Add points as scatter
    fig.add_trace(go.Scatter(
        x=points_df['x'],
        y=points_df['y'],
        mode='markers+text',
        text=points_df['name'],
        textposition='top center',
        marker=dict(size=10, color='#1f77b4', symbol='circle')
    ))

    # Set title and labels
    fig.update_layout(
        title="Solution path",
        title_x=0.5,
        xaxis_title="X coordinate",
        yaxis_title="Y coordinate",
        showlegend=False,
        template="plotly_white"
    )

    # Show the plot
    return fig


def solutions_over_time_plot(points_df, solutions):
    """
    Creates an animated plot showing the solutions to the Traveling Salesman Problem over time.

    Args:
    - points_df (pandas.DataFrame): A dataframe containing the points with 'name', 'x', and 'y' columns.
    - solutions (numpy.ndarray): A 2D array where each row is a solution with indices of the path.

    Returns:
    - plotly.graph_objects.Figure: The animated figure showing the solutions.
    """
    
    # Function to create data for each frame
    def create_frame(solution):
        # Extracting the coordinates of the points according to the solution
        x_vals = points_df['x'].iloc[solution].values
        y_vals = points_df['y'].iloc[solution].values
        
        # Creating the lines (path)
        path_x = np.concatenate([x_vals, [x_vals[0]]])  # Closing the cycle of the salesman
        path_y = np.concatenate([y_vals, [y_vals[0]]])  # Closing the cycle of the salesman
        
        # Creating the plot for the frame
        scatter = go.Scatter(
            x=points_df['x'], 
            y=points_df['y'], 
            mode='markers+text', 
            text=points_df['name'], 
            textposition='top center',
            marker=dict(size=10, color='#1f77b4'),
            showlegend=False
        )
        
        path = go.Scatter(
            x=path_x, 
            y=path_y, 
            mode='lines', 
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        )
        
        return [path, scatter]

    # Creating the animation with frames
    frames = [go.Frame(data=create_frame(solution), name=f'Frame {i}') for i, solution in enumerate(solutions)]

    # Layout of the plot
    layout = go.Layout(
        title='Evolution of the solution throughout the optimization epochs',
        title_x=0.5,
        xaxis={'title': 'X coordinate'},
        yaxis={'title': 'Y coordinate'},
        template='plotly_white',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            }],
            'x': 0,  # Placing the "Play" button more to the left
            'xanchor': 'left',
            'y': -0.12,  # Placing the button below the plot
            'yanchor': 'top'
        }],
        sliders=[{
            'currentvalue': {'visible': True, 'prefix': 'Epoch: ', 'font': {'size': 15, 'color': 'black'}},
            'steps': [{
                'label': str(i), 
                'method': 'animate', 
                'args': [[f'Frame {i}'], {'mode': 'immediate', 'frame': {'duration': 500}}]
            } for i in range(len(solutions))],
            'pad': {'b': 50},  # Adjusting the space below the slider
            'x': 0.1,
            'len': 0.9
        }]
    )

    # Creating the figure
    fig = go.Figure(
        data=create_frame(solutions[0]),  # Using the first solution to start the plot
        layout=layout,
        frames=frames
    )

    return fig


def pheromone_distribution_plot(points_df, pheromone_matrices):
    """
    Creates an animated plot showing the pheromone distribution over iterations of the ACO algorithm.

    Args:
    - points_df (pandas.DataFrame): A dataframe containing the points with 'name', 'x', and 'y' columns.
    - pheromone_matrices (list of numpy.ndarray): A list where each element is a matrix representing the pheromone levels
      for each edge (path between two points) at a given iteration.

    Returns:
    - plotly.graph_objects.Figure: The animated figure showing the pheromone distribution.
    """
    pheromone_max = np.max(pheromone_matrices)

    # Function to create data for each frame
    def create_frame(pheromone_matrix):
        # Creating the lines (paths) based on pheromone levels
        lines = []
        
        for i, point_i in points_df.iterrows():
            for j, point_j in points_df.iterrows():
                if i < j:  # Only drawing each edge once
                    pheromone_level = pheromone_matrix[i, j]
                    # Calculating the alpha component (opacity) based on pheromone level
                    alpha = min(pheromone_level / pheromone_max, 1.0)  # Normalize to [0, 1]
                    
                    # Extract the coordinates of the points
                    x_vals = [point_i['x'], point_j['x']]
                    y_vals = [point_i['y'], point_j['y']]
                    
                    # Create the line with the corresponding pheromone level
                    line = go.Scatter(
                        x=x_vals, 
                        y=y_vals, 
                        mode='lines',
                        line=dict(color=f'rgba(255, 0, 0, {alpha})', width=2),
                        showlegend=False
                    )
                    lines.append(line)
        
        return lines

    # Creating the animation with frames
    frames = [go.Frame(data=create_frame(pheromone_matrix), name=f'Frame {i}') 
              for i, pheromone_matrix in enumerate(pheromone_matrices)]

    # Layout of the plot
    layout = go.Layout(
        title='Pheromone distribution across optimization epochs',
        title_x=0.5,
        xaxis={'title': 'X'},
        yaxis={'title': 'Y'},
        template='plotly_white',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            }],
            'x': 0,  # Placing the "Play" button more to the left
            'xanchor': 'left',
            'y': -0.15,  # Placing the button below the plot
            'yanchor': 'top'
        }],
        sliders=[{
            'currentvalue': {'visible': True, 'prefix': 'Epoch: ', 'font': {'size': 15, 'color': 'black'}},
            'steps': [{
                'label': str(i), 'method': 'animate', 'args': [[f'Frame {i}'], {'mode': 'immediate', 'frame': {'duration': 500}}]
            } for i in range(len(pheromone_matrices))],
            'pad': {'b': 50},  # Adjusting the space below the slider
            'x': 0.1,
            'len': 0.9
        }]
    )

    # Creating the figure
    fig = go.Figure(
        data=create_frame(pheromone_matrices[0]),  # Using the first pheromone matrix to start the plot
        layout=layout,
        frames=frames
    )

    return fig



def length_over_time_plot(best_length):
    """
    Create a line plot displaying the evolution of the best path length over time.

    Parameters:
    - best_length (list): List of the best path lengths over epochs.

    Returns:
    - fig (plotly.graph_objects.Figure): Plotly figure with the line plot.
    """
    # Create a figure with a line for the best path length
    fig = go.Figure()

    # Add best path line
    fig.add_trace(go.Scatter(
        x=list(range(len(best_length))),
        y=best_length,
        mode='lines'
    ))

    # Customize layout
    fig.update_layout(
        title='Solution path length across optimization epochs',
        title_x=0.5,
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Length'),
        template='plotly_white',  # Apply plotly_white theme
    )

    return fig


def entropy_over_time_plot(pheromone_matrices):
    """
    Plot the entropy of the pheromone distribution over time (epochs).
    
    Parameters:
    - pheromone_matrices (list): List of pheromone matrices for each epoch (numpy arrays).
    
    Returns:
    - fig (plotly.graph_objects.Figure): Plotly figure with the entropy evolution.
    """
    # List to store entropy values for each epoch
    entropy_values = []
    
    # Compute entropy for each pheromone matrix
    for pheromone_matrix in pheromone_matrices:
        # Flatten the matrix and normalize it (to avoid zero probabilities which can cause errors in entropy calculation)
        pheromone_flat = pheromone_matrix.flatten()
        pheromone_flat = pheromone_flat / np.sum(pheromone_flat)  # Normalize to get probability distribution
        
        # Compute the entropy of the flattened pheromone matrix
        entr = entropy(pheromone_flat)
        entropy_values.append(entr)
    
    # Create the figure with the entropy evolution plot
    fig = go.Figure()

    # Add entropy plot as a line graph
    fig.add_trace(go.Scatter(
        x=list(range(len(entropy_values))),
        y=entropy_values,
        mode='lines',
        line=dict(color='blue', width=2)
    ))

    # Customize layout
    fig.update_layout(
        title='Entropy of pheromone distribution over time',
        title_x=0.5,
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Entropy'),
        template='plotly_white',  # Apply plotly_white theme
        showlegend=True
    )

    return fig