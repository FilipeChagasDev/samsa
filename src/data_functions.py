'''
Filipe Chagas Ferraz (github.com/FilipeChagasDev)
Nov-2024
'''
import pandas as pd
import numpy as np
import tempfile
import os


def euclidean_distance_matrix_memmap(points_df: pd.DataFrame) -> np.ndarray:
    """
    Computes the Euclidean distance matrix for a set of points, using a temporary memory-mapped file
    to reduce RAM usage. The temporary file is deleted automatically after use.

    Parameters:
        points_df (pd.DataFrame): A DataFrame with columns 'x' and 'y' representing the coordinates of the points.
    
    Returns:
        np.ndarray: A memory-mapped 2D NumPy array (matrix) where the element at position (i, j) is the Euclidean distance between point i and point j.
    """
    coords = points_df[['x', 'y']].values
    n = coords.shape[0]
    
    # Criação de um arquivo temporário
    tmp_file = tempfile.NamedTemporaryFile()
    
    # Criação de um array de memória mapeada para a matriz de distâncias
    dist_matrix = np.memmap(tmp_file.name, dtype='float32', mode='w+', shape=(n, n))
    
    # Preenchendo a matriz de distâncias
    for i in range(n):
        for j in range(i+1, n):  # Só precisa calcular a metade superior (simétrica)
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # A matriz é simétrica

    return dist_matrix, tmp_file


def euclidean_distance_matrix(points_df: pd.DataFrame, use_memmap: bool = True) -> np.ndarray:
    """
    Computes the Euclidean distance matrix for a set of points.

    This function takes a DataFrame containing the coordinates (x, y) of points and calculates the 
    pairwise Euclidean distance between each pair of points. The result is a square matrix where 
    the element at position (i, j) represents the Euclidean distance between point i and point j.

    Parameters:
        points_df (pd.DataFrame): A DataFrame with columns 'x' and 'y' representing the coordinates of the points.
    
    Returns:
        np.ndarray: A 2D NumPy array (matrix) where the element at position (i, j) is the Euclidean distance between point i and point j.
    """ 
    coords = points_df[['x', 'y']].values
    return np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)


def distance_matrix_from_df(points_df: pd.DataFrame, distances_df: pd.DataFrame) -> np.ndarray:
    """
    Creates an adjacency matrix from a DataFrame of pairwise distances between points.

    This function takes two DataFrames: one containing the points and their coordinates, and another 
    containing the pairwise distances between points. It returns a square adjacency matrix where the 
    element at position (i, j) represents the distance between point i and point j. If no distance 
    is provided between two points, the value will be infinity (np.inf), indicating no direct connection.

    Parameters:
        points_df (pd.DataFrame): A DataFrame containing the points with their 'name' and coordinates ('x', 'y').
        distances_df (pd.DataFrame): A DataFrame containing 'origin', 'destination', and 'distance' columns.
                                    'origin' and 'destination' are point names, and 'distance' is the pairwise distance between them.
    
    Returns:
        np.ndarray: A 2D NumPy array (adjacency matrix) where the element at position (i, j) is the distance 
                    between points i and j. If no direct distance is provided, the value will be np.inf.
    """
    # Create a dictionary to map point names to indices (positions in the adjacency matrix)
    index_map = {name: idx for idx, name in enumerate(points_df['name'])}

    # Initialize the adjacency matrix with np.inf (representing "infinite" distance for non-adjacent points)
    n = len(points_df)  # Number of points
    adj_matrix = np.full((n, n), np.inf)  # Create a square matrix filled with np.inf

    # Fill the adjacency matrix with distances from df_distancias
    # Loop through each row in the distance DataFrame to fill the adjacency matrix
    for _, row in distances_df.iterrows():
        # Get the indices of the points from the index_map
        idx_a = index_map[row['origin']]
        idx_b = index_map[row['destination']]
        
        # Fill both (i, j) and (j, i) positions in the matrix (since adjacency is bidirectional)
        adj_matrix[idx_a, idx_b] = row['distance']
        adj_matrix[idx_b, idx_a] = row['distance']

    return adj_matrix


def indices_to_names(indices, df_pontos):
    """
    Converts a sequence of indices from the adjacency matrix into a sequence of point names.
    
    Parameters:
    indices (list or array-like): A sequence of indices corresponding to the points in the adjacency matrix.
    df_pontos (pandas.DataFrame): The DataFrame containing the points and their names.
    
    Returns:
    list: A list of point names corresponding to the given indices.
    """
    # Step 1: Create a mapping of indices to point names from the 'name' column of df_pontos
    name_map = df_pontos['name'].to_dict()
    
    # Step 2: Use the indices to retrieve the corresponding point names
    point_names = [name_map[idx] for idx in indices]
    
    return point_names