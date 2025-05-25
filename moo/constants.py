from botorch.test_functions.multi_objective import *

N = 10  # Number of runs per acquisition function
FIG_DIR = "figures" 
NUMERICAL_RESULTS_DIR = "numerical_results"
# Define list function tests
OBJECTIVE_FUNCTIONS = {
    "BraninCurrin": BraninCurrin,
    "DTLZ1": DTLZ1,
    "DTLZ2": DTLZ2,
    "DTLZ3": DTLZ3,
    "DTLZ4": DTLZ4,
    "DTLZ5": DTLZ5,
    "DTLZ7": DTLZ7,
    "Penicillin": Penicillin, # works
    "VehicleSafety": VehicleSafety, # works
    "ZDT1": ZDT1, # works
    "ZDT2": ZDT2, # works
    "ZDT3": ZDT3, # works
    "CarSideImpact": CarSideImpact, # works
}
FUNCTION_ARGS = {
    BraninCurrin: {},
    DTLZ1: {"dim": 6, "num_objectives": 2, "negate": True},
    DTLZ2: {"dim": 6, "num_objectives": 2, "negate": True},
    DTLZ3: {"dim": 6, "num_objectives": 2, "negate": True},
    DTLZ4: {"dim": 6, "num_objectives": 2, "negate": True},
    DTLZ5: {"dim": 6, "num_objectives": 2, "negate": True},
    DTLZ7: {"dim": 6, "num_objectives": 2, "negate": True},
    Penicillin: {"negate": True},
    VehicleSafety: {"negate": True},
    ZDT1: {"dim": 5, "negate": True},
    ZDT2: {"dim": 5, "negate": True},
    ZDT3: {"dim": 5, "negate": True},
    CarSideImpact: {"negate": True},
}
MC_SAMPLES = 128  # Number of Monte Carlo samples for acquisition function
BETA = 20
NOISE_SE = 0.1