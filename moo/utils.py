import os
import numpy as np

def count_results(file_name):
    """
    Count the number of results in a given numpy file.
    
    Args:
        file_name (str): The name of the file to count results in.
        
    Returns:
        int: The number of results in the file.
    """
    if os.path.exists(file_name):
        data = np.load(file_name)
        return data.shape[0]
    else:
        print(f"File {file_name} does not exist.")
        return 0
    
def check_results(file_name):
    """
    Check if the results in a given numpy file are enough.
    
    Args:
        file_name (str): The name of the file to check results in.
        
    Returns:
        bool: True if the results are empty, False otherwise.
    """
    return count_results(file_name)==10

def save_results_to_existing_file(file_name, results):
    """
    Save results to an existing numpy file.
    
    Args:
        file_name (str): The name of the file to save results in.
        results (np.ndarray): The results to save.
    """
    if os.path.exists(file_name):
        data = np.load(file_name)
        data = np.concatenate((data, results), axis=0)
        np.save(file_name, data)
    else:
        np.save(file_name, results)