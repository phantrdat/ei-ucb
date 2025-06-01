from botorch.test_functions import *
from test_functions.test_functions import *
# from test_functions.lunar_lander import LunarLander
FIG_DIR = "res_figures" 
NUMERICAL_RESULTS_DIR= "/home/pdat/EI-UCB/sbo/numerical_results" 
EXP_RUNS  = 10  # Number of runs per acquisition function
OBJECTIVE_FUNCTIONS = [
    # Ackley,
    # Beale,
    Branin,
    # Bukin,
        DixonPrice,
    # DropWave,
    # EggHolder,
        Griewank,
        Alpine1,
    # Hartmann,
    # HolderTable,
        Levy,
    # Michalewicz,
        Powell,
        Rastrigin,
        Rosenbrock,
    # Shekel,
    # SixHumpCamel,
        StyblinskiTang
    ]
DIMS = {
    Branin: 2,
    Ackley: 50,
    Alpine1: 15,
    DixonPrice: 15,
    Griewank:9,
    Levy: 13,
    Powell: 18,
    Rastrigin: 23,
    Rosenbrock: 24,
    StyblinskiTang: 21
}

# DIMS = {
#     Ackley: 200,
#     Alpine1: 200,
#     DixonPrice: 200,
#     Griewank: 200,
#     Levy: 200,
#     Powell: 200,
#     Rastrigin: 200,
#     Rosenbrock: 200,
#     StyblinskiTang: 200
# }