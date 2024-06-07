import numpy as np

def minimal_distance(list1, list2, full_scale=360):
    if len(list1) == 0 or len(list2) == 0:
        return full_scale
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Determine the sizes of the lists
    len1 = len(array1)
    len2 = len(array2)
    
    error_matrix = np.empty((len1, len2))
    for i in range(len1):
        error_matrix[i] = np.abs(array1[i] - array2)
        error_matrix[i] = good_error(array1[i], array2, full_scale)
    # Find the minimum error
    min_error = np.mean(np.min(error_matrix, axis=1))
    return min_error
def good_error(a, b, full_scale=360):
    abs_error = np.abs(a - b)
    error = np.minimum(abs_error, full_scale - abs_error)
    return error
