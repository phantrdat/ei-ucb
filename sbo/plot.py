import numpy as np
from matplotlib import pyplot as plt
from constants import OBJECTIVE_FUNCTIONS, NUMERICAL_RESULTS_DIR, FIG_DIR, DIMS
import os
baselines = ['ei', 'logei', 'qlogei', 'turbo']
colors = {'ei': 'blue',
    'turbo': 'cyan',
    'logei': 'green',
    'qlogei': 'purple',
    'custom_ei': 'red'}
markers = {'ei': 'o',
    'turbo': 's',
    'logei': 'x',
    'qlogei': 'd',
    'custom_ei': '^'}  
SCALE = 0.5
plt.figure(figsize=(8, 5))
for f in OBJECTIVE_FUNCTIONS[:1]:
    func_name = f().__class__.__name__
    dim = DIMS[f]
    for beta in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
        data_path = f"{NUMERICAL_RESULTS_DIR}/{func_name}_{dim}/{func_name}_{dim}_best_values_ei_custom_beta={beta}.npy"
        if not os.path.exists(data_path):
            print(f"Data file {data_path} does not exist. Skipping.")
        else:
            ei_custom_data = np.load(data_path)
            mean_ei_custom = ei_custom_data.mean(axis=0)
            std_ei_custom = ei_custom_data.std(axis=0)
            iterations = np.arange(len(mean_ei_custom))
            plt.plot(iterations, mean_ei_custom, marker=markers['custom_ei'], linestyle="-", color=colors['custom_ei'], label=f"EI Custom, Beta={beta}")
            plt.fill_between(iterations, mean_ei_custom - SCALE*std_ei_custom, mean_ei_custom + SCALE*std_ei_custom, color=colors['custom_ei'], alpha=0.2)

        for b in baselines: 
            print(b)
            data_path = f"{NUMERICAL_RESULTS_DIR}/{func_name}_{dim}/{func_name}_{dim}_best_values_{b}.npy"
            if not os.path.exists(data_path):
                print(f"Data file {data_path} does not exist. Skipping.")
                
            else:
                best_values = np.load(data_path)
                mean_values = best_values.mean(axis=0)
                std_values  = best_values.std(axis=0)
                iterations  = np.arange(len(mean_values))

                plt.plot(iterations, mean_values, marker=markers[b], linestyle="-", label=f"{b} (mean)")
                plt.fill_between(iterations, mean_values - SCALE*std_values, mean_values + SCALE*std_values, alpha=0.2)
        plt.xlabel("Iterations")
        plt.ylabel("Best Value")
        plt.title(f"{func_name} - Comparison of Acquisition Functions")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if os.path.isdir(f"{FIG_DIR}/{func_name}_{dim}") == False:
            os.makedirs(f"{FIG_DIR}/{func_name}_{dim}")
        plt.savefig(f"{FIG_DIR}/{func_name}_{dim}/{func_name}_{dim}-beta={beta}.pdf", dpi=300)  # Save plot as PDF
        plt.clf()  # Clear the figure for the next beta value

## -------------------------------------------------------------------------------------------------------------
# SCALE = 0.5
# baselines = ['turbo-ts', 'turbo-ei', 'turbo-custom_ei']
# markers = {'turbo-ts': 'o', 'turbo-ei': 's',    'turbo-custom_ei': 'd'}
# colors = {'turbo-ts': 'blue', 'turbo-ei': 'red', 'turbo-custom_ei': 'green'}
# plt.figure(figsize=(8, 5))


# for f in OBJECTIVE_FUNCTIONS:
#     func_name = f().__class__.__name__
#     dim = DIMS[f]
#     for b in baselines: 
#         print(b)
#         data_path = f"{NUMERICAL_RESULTS_DIR}/{func_name}_{dim}/{func_name}_{dim}_best_values_{b}.npy"
#         if not os.path.exists(data_path):
#             print(f"Data file {data_path} does not exist. Skipping.")
#             continue
#         best_values = np.load(data_path)
#         mean_values = best_values.mean(axis=0)
#         std_values = best_values.std(axis=0)
#         iterations = np.arange(len(mean_values))

#         plt.plot(iterations, mean_values, marker=markers[b], linestyle="-", label=f"{b} (mean)")
#         plt.fill_between(iterations, mean_values - SCALE*std_values, mean_values + SCALE*std_values, alpha=0.2)
#     plt.xlabel("Iterations")
#     plt.ylabel("Best Value")
#     plt.title(f"{func_name} - Comparison of Acquisition Functions")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     if os.path.isdir(f"{FIG_DIR}/{func_name}") == False:
#         os.makedirs(f"{FIG_DIR}/{func_name}")
#     plt.savefig(f"{FIG_DIR}/{func_name}/{func_name}-{dim}.pdf", dpi=300)  # Save plot as PDF
#     plt.clf()  # Clear the figure for the next beta value

