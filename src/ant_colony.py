'''
Filipe Chagas Ferraz (github.com/FilipeChagasDev)
Nov-2024
'''
import numpy as np


class AntColonyOptimizer():
    def __init__(self, distance_matrix: np.ndarray, n_ants: int, n_epochs: int, alpha: float, beta: float, rho: float, zq: float):
        # Params:
        #   distance_matrix: Numpy 2D array where each distance_matrix[i,j] is the distance between the i-th and j-th points.
        #   n_ants: number of ants walking together on each epoch.
        #   n_epochs: number of optimization epochs/iterations.
        #   alpha and beta: hyperparameters of the move probability distribution.
        #   rho: pheromone evaporation rate.
        assert n_ants > 0
        assert n_epochs > 0
        assert distance_matrix.shape[0] == distance_matrix.shape[1]
        assert distance_matrix.shape[0] > 1
        self.n_ants = n_ants
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.distance_matrix = distance_matrix
        self.symmetric = np.all(distance_matrix == distance_matrix.T) # True if the graph is not directed
        self.n_points = distance_matrix.shape[0]
        self.q = np.mean(distance_matrix)*self.n_points - zq*np.std(distance_matrix)*np.sqrt(self.n_points)
        self.pheromone = np.ones(shape=(self.n_points, self.n_points))
        self.best_path = None
        self.best_length = float('inf')
        self.worst_length = float('-inf')
    

    def __move_probs(self, visited: np.ndarray, curr_point: int) -> np.ndarray:
        # Method that returns the probability distribution of moving from a current point to each unvisited point
        # Params:
        #   visited: Numpy 1D array of booleans. If visited[i] is True, then the i-th point is already visited.
        #   curr_point: index of the current point.
        assert 0 <= curr_point < self.n_points
        assert visited.dtype == bool
        probs = np.zeros(self.n_points)
        for j in range(self.n_points):
            if not visited[j]:
                probs[j] = self.pheromone[curr_point, j]**self.alpha * (1/self.distance_matrix[curr_point, j])**self.beta
        probs /= np.sum(probs)
        return probs
    

    def __path_length(self, path: np.ndarray) -> float:
        # Calculates the length of a given path.
        # Params:
        #   path: Numpy 1D array of point indexes.
        assert path.shape == (self.n_points,)
        length = 0
        for i in range(self.n_points-1):
            length += self.distance_matrix[path[i], path[i+1]]
        length += self.distance_matrix[path[-1], path[0]]
        return length


    def __ant_walk(self) -> tuple:
        # Simulates the walk of a single ant.
        # Returns the path walked by the ant and it's length.
        visited = np.zeros(self.n_points, dtype=bool)
        path = np.zeros(self.n_points, dtype=int)
        path[0] = np.random.randint(0, self.n_points)
        visited[path[0]] = True
        for i in range(1, self.n_points):
            probs = self.__move_probs(visited, path[i-1])
            chosen_point = np.random.choice(np.arange(self.n_points), p=probs)
            path[i] = chosen_point
            visited[chosen_point] = True
        return path, self.__path_length(path)
    

    def __update_pheromone(self):
        # Method that updates the pheromone distribution.
        assert self.best_path is not None

        self.pheromone *= 1 - self.rho # Evaporate pheromone
        
        for i in range(self.n_points-1):
            self.pheromone[self.best_path[i], self.best_path[i+1]] += self.q/self.best_length
            if self.symmetric:
                self.pheromone[self.best_path[i+1], self.best_path[i]] = self.pheromone[self.best_path[i], self.best_path[i+1]]
        
        self.pheromone[self.best_path[-1], self.best_path[0]] += self.q/self.best_length
        if self.symmetric:
            self.pheromone[self.best_path[0], self.best_path[-1]] = self.pheromone[self.best_path[-1], self.best_path[0]]


    def __optim_step(self):
        # Perform a single optimization epoch.
        for k in range(self.n_ants):
            path, length = self.__ant_walk()
            if self.best_length > length:
                self.best_length = length
                self.best_path = path
            if self.worst_length < length:
                self.worst_length = length
        self.__update_pheromone()
    

    def optimize(self, path_history: list = None, best_length_history: list = None, worst_length_history: list = None, pheromone_history: list = None):
        # Perform the complete optimization process. 
        # Params:
        #   path_history: list where the best path at each epoch is added.
        #   length_history: list where the best length at each epoch is added.
        #   pheromone_history: list where the pheromone distribution matrix at each epoch is added.
        for i in range(self.n_epochs):
            self.__optim_step()
            
            if path_history is not None:
                path_history.append(self.best_path.copy())

            if best_length_history is not None:
                best_length_history.append(self.best_length.copy())

            if worst_length_history is not None:
                worst_length_history.append(self.worst_length.copy())

            if pheromone_history is not None:
                pheromone_history.append(self.pheromone.copy())

        return self.best_path, self.best_length