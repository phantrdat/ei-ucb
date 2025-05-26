from botorch.test_functions import *
from test_functions.test_functions import *
# from test_functions.lunar_lander import LunarLander
FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR= "numerical_results" 
EXP_RUNS  = 30  # Number of runs per acquisition function
OBJECTIVE_FUNCTIONS = [
    Ackley,
    # Beale,
    # Branin,
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
    Ackley: 20,
    Alpine1: 15,
    DixonPrice: 15,
    Griewank:9,
    Levy: 13,
    Powell: 18,
    Rastrigin: 23,
    Rosenbrock: 24,
    StyblinskiTang: 21
}