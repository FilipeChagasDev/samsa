help_article = '''
# SAMSA Instruction Manual

SAMSA is an educational and non-commercial app that can be used to demonstrate the Traveling Salesman Problem (TSP) 
and its solution process using the Ant Colony Optimization (ACO) metaheuristic algorithm. In this app, the user can 
input or generate a set of entry points, configure the ACO parameters, and the app will handle finding and presenting 
the solution to the TSP instance. Once the solution is found, it is possible to visualize the path taken by the 
traveling salesman, as well as the evolution of the ACO states throughout the optimization epochs.

This manual provides a brief explanation of the Traveling Salesman Problem and the Ant Colony Optimization algorithm. 
It then covers details about the SAMSA interface and its outputs.

## Traveling Salesman Problem

The Traveling Salesman Problem (TSP) is a classic optimization problem where a salesman must visit a set of 
cities/points $\{p_1, p_2, ..., p_n\}$ exactly once and return to the starting point, while minimizing the 
total travel distance or cost. As the number of cities increases, the number of possible routes grows exponentially, 
making it computationally difficult to find the optimal solution. Since TSP is an NP-hard problem, exact solutions 
are often impractical for large instances. To tackle this, metaheuristic algorithms are commonly used, providing 
approximate solutions in a reasonable amount of time. One such algorithm is Ant Colony Optimization (ACO), which 
mimics the behavior of ants in finding the shortest path to solve the TSP. ACO, along with other metaheuristics, 
offers a powerful approach for finding good solutions to the problem, even if they are not necessarily optimal.

## Ant Colony Optimization 

Ant Colony Optimization (ACO) is a metaheuristic originally introduced by Marco Dorigo in 1992, inspired by the foraging 
behavior of ants. In nature, ants deposit pheromones on their paths as they search for food, and other ants follow 
these pheromone trails, with stronger trails attracting more ants. Over time, shorter paths accumulate more pheromones 
and are more likely to be chosen, leading to an efficient solution. In ACO, artificial "ants" simulate this behavior 
to explore the solution space and iteratively improve upon possible solutions. It is particularly effective for solving 
combinatorial optimization problems, such as the Traveling Salesman Problem (TSP), by gradually converging to an optimal 
or near-optimal solution. ACO is known for its ability to balance exploration (trying new paths) and exploitation 
(refining known paths) through the use of pheromone updating rules and heuristic information.

### Parameters
* **Number of Ants** - This parameter determines how many artificial ants are used in the search process during each iteration. Each ant represents a potential solution and explores the problem space by selecting paths based on pheromone levels and heuristic information. A larger number of ants generally increases the diversity of solutions explored, potentially leading to better results, but also increases computational cost.
* **Number of epochs** - his parameter refers to the number of iterations or cycles the algorithm will perform. During each epoch, ants construct solutions and update pheromones based on the quality of the solutions. The algorithm typically runs for multiple epochs to refine the solutions and converge towards an optimal or near-optimal result. More epochs can lead to more accurate solutions but also require more computational time.
* **α (alpha)** - Alpha controls the influence of pheromone levels in the decision-making process. It determines the weight given to the pheromone trail when ants choose their paths. A higher value of alpha means the algorithm will be more reliant on pheromone trails, promoting exploitation of known good paths. A lower value encourages exploration by reducing the influence of pheromones in the path selection.
* **β (beta)** - Beta controls the influence of heuristic information (such as distance or cost) in the path selection process. It determines how much importance is given to the problem’s inherent features, like the distance between cities in the TSP. A higher beta value makes the algorithm focus more on the heuristic information, guiding ants toward more promising paths. A lower value makes ants rely more on pheromone information.
* **ρ (rho)** - Rho represents the pheromone evaporation rate. It defines how quickly the pheromone trail decays over time. A higher rho value causes pheromones to evaporate faster, which helps to avoid stagnation and encourages exploration of new paths. A lower value means pheromones persist longer, promoting exploitation of successful paths but possibly causing the algorithm to get stuck in local optima.
* **ζ (zeta)** - In this particular implementation, zeta determines the z-score of the amount of pheromone deposited in each optimization epoch. This z-score is related to the mean and standard deviation of the lengths of all possible solutions to the problem.

## SAMSA Interface

### Sidebar

In the sidebar, there are two input sections: **Pointset** and **Ant Colony Optimization Parameters**.

#### Pointset section

In the Pointset section, you have the options to generate a random set of points for the traveling salesman problem 
or upload a CSV file containing the points. The CSV file should include the following columns:

* **name** - a label for the point
* **x** - horizontal coordinate of the point
* **y** - vertical coordinate of the point

You can provide a CSV with latitudes and longitudes of geographic points, but you should be aware that SAMSA calculates 
Euclidean (flat) distances between points. Since the Earth is not flat, the path distance provided as a result will not 
be the geodesic distance in meters or kilometers. You should also keep in mind that, in practice, it is desirable to 
consider factors such as road sizes and travel time, but SAMSA does not account for either of these.

### Ant Colony Optimization Parameters

In this section, you should enter the values for the parameters explained above, which are: number of ants, number of epochs, alpha, beta, rho, and zeta.

### Problem instance tab

### Solution tab

### Solution over time tab

### Pheromone over time tab

'''