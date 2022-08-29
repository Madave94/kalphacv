import krippendorff
import math
import numpy as np

def calculate_alpha(coincidence_matrix):
    """
    calculate krippendorff's alpha of the coincidence matrix classes
    """
    processed_coincidence_class_matrix = convert_to_classes(coincidence_matrix)
    num_classes = check_num_classes(processed_coincidence_class_matrix)
    processed_coincidence_class_matrix = handle_krippendorff_special_case(processed_coincidence_class_matrix, num_classes)
    if num_classes > 1:
        alpha = round(krippendorff.alpha(processed_coincidence_class_matrix, level_of_measurement="nominal"), 4)
    else:
        alpha = 1
    if math.isnan(alpha): # if alpha returns nan, then there was no annotation from all but one annotator
        alpha = 0
    return alpha

def handle_krippendorff_special_case(processed_coincidence_class_matrix, num_classes):
    """
    handle special case: if coincidence matrix contains one class or is non
    """
    if num_classes == 2 and np.any(np.isnan(processed_coincidence_class_matrix)): # special case for only nan and one class
        processed_coincidence_class_matrix[np.isnan(processed_coincidence_class_matrix)] = 0
    return processed_coincidence_class_matrix

def convert_to_classes(coincidence_matrix):
    """
    @param coincidence_matrix: coincidence matrix containing matrix entries
    @return: coincidence matrix containing only classes

    """
    coincidence_class_matrix = np.zeros((len(coincidence_matrix), len(coincidence_matrix[0])))
    for i in range(0, len(coincidence_matrix)):
        for j in range(0, len(coincidence_matrix[i])):
            if coincidence_matrix[i][j].category == "*":
                coincidence_class_matrix[i][j] = 0 # if this is np.nan don't punish missing boxes
            else:
                coincidence_class_matrix[i][j] = coincidence_matrix[i][j].category
    return coincidence_class_matrix

def check_num_classes(coincidence_class_matrix):
    """
        check the number of classes in the coincidence matrix to escape error in krippendorff calculation
    """
    num_classes = np.unique(coincidence_class_matrix)
    return len(num_classes)
