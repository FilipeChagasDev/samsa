# 𖢥 SAMSA


SAMSA is an educational and non-commercial app that can be used to demonstrate the Traveling Salesman Problem (TSP) 
and its solution process using the Ant Colony Optimization (ACO) metaheuristic algorithm. In this app, the user can 
input or generate a set of entry points, configure the ACO parameters, and the app will handle finding and presenting 
the solution to the TSP instance. Once the solution is found, it is possible to visualize the path taken by the 
traveling salesman, as well as the evolution of the ACO states throughout the optimization epochs.

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

## Deploy command

```sh
gunicorn --chdir src app:server
```