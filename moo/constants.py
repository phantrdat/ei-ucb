from botorch.test_functions.multi_objective import *

N = 10  # Number of runs per acquisition function
FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR = "numerical_results"
# Define list function tests
OBJECTIVE_FUNCTIONS = [
    # BraninCurrin,
    # DH1,
    # DH2,
    # DH3,
    # DH4,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ7,
    Penicillin,
    VehicleSafety,
    ZDT1,
    ZDT2,
    ZDT3,
    CarSideImpact,
]
FUNCTION_ARGS = {
    # BraninCurrin: {"negate": True},
    # DH1: {"dim": 5},
    # DH2: {"dim": 5},
    # DH3: {"dim": 5},
    # DH4: {"dim": 5},
    DTLZ1: {"dim": 6, "num_objectives": 2},
    DTLZ2: {"dim": 6, "num_objectives": 2},
    DTLZ3: {"dim": 6, "num_objectives": 2},
    DTLZ4: {"dim": 6, "num_objectives": 2},
    DTLZ5: {"dim": 6, "num_objectives": 2},
    DTLZ7: {"dim": 6, "num_objectives": 2},
    Penicillin: {},
    VehicleSafety: {},
    ZDT1: {"dim": 5},
    ZDT2: {"dim": 5},
    ZDT3: {"dim": 5},
    CarSideImpact: {},
}
MC_SAMPLES = 16  # Number of Monte Carlo samples for acquisition function
BETA = 20
NOISE_SE = 0.1