import numpy as np
from scipy.optimize import linear_sum_assignment

from iaadet import reliability_data
from iaadet import iou

def run_matching(available_entries_old_ann, unmatched_entries_new_ann, iou_threshold, mode):
    """
        Function to extract elements from the entries, run the hungarian matching and
        return all possible matches as a list of tuples in the following form:
        [(entry_1_ann_A, entry_1_ann_B), (entry_2_ann_A, entry_2_ann_B_) ... ]

        Elements that have not been matched are not mentioned here any further since we can find
        them easily via set operations
    """

    filtered_entries_old_ann, filtered_entries_new_ann = filter_matching_entries(list(available_entries_old_ann),
                                                                                 list(unmatched_entries_new_ann),
                                                                                 iou_threshold, mode)

    if len(filtered_entries_old_ann) < 1 or len(filtered_entries_new_ann) < 1:
        return []

    return hungarian_algorithm(filtered_entries_old_ann, filtered_entries_new_ann, iou_threshold, mode)


def filter_matching_entries(available_entries_old_ann, unmatched_entries_new_ann, iou_threshold, mode):
    """
        add all entries that have a matching entry with another annotator to matching_entries_old_ann and
        matching_entries_new_ann
    """
    matching_entries_old_ann = []
    matching_entries_new_ann = []
    iou_matrix = generate_iou_matrix(available_entries_old_ann, unmatched_entries_new_ann, mode)

    for row in range(0, len(iou_matrix)):
        if is_in_threshold(iou_matrix[row], iou_threshold):
            matching_entries_old_ann.append(available_entries_old_ann[row])
    for col in range(0, len(iou_matrix.T)):
        if is_in_threshold(iou_matrix.T[col], iou_threshold):
            matching_entries_new_ann.append(unmatched_entries_new_ann[col])

    if len(matching_entries_old_ann) != len(matching_entries_new_ann):
        matching_entries_old_ann, matching_entries_new_ann = add_empty_entries(matching_entries_old_ann,
                                                                               matching_entries_new_ann)

    return matching_entries_old_ann, matching_entries_new_ann


def is_in_threshold(iou_list, iou_threshold):
    """
    check if one entry matches with another entry
    """
    min_value = min(iou_list)
    if min_value < 1 - iou_threshold:
        return True
    else:
        return False


def add_empty_entries(old_ann, new_ann):
    num_empty_entries = abs(len(old_ann) - len(new_ann))
    if len(old_ann) < len(new_ann):
        annotator_name = old_ann[0].annotator_name
        old_ann.extend([reliability_data.EmptyEntry(annotator_name) for _ in range(num_empty_entries)])
    else:
        annotator_name = new_ann[0].annotator_name
        new_ann.extend([reliability_data.EmptyEntry(annotator_name) for _ in range(num_empty_entries)])
    return old_ann, new_ann


def hungarian_algorithm(available_entries_old_ann, unmatched_entries_new_ann, iou_threshold, mode):
    """
        hungarian matching algorithm
    """
    matching_list = []
    hungarian_matching_matrix = generate_iou_matrix(available_entries_old_ann, unmatched_entries_new_ann, mode)
    row_ind, col_ind = linear_sum_assignment(hungarian_matching_matrix)

    # add bounding box pair to matching list, if their iou value is larger than the threshold
    for i in range(0, len(row_ind)):
        if hungarian_matching_matrix[row_ind[i]][col_ind[i]] < 1 - iou_threshold:
            matching_list.append((available_entries_old_ann[row_ind[i]], unmatched_entries_new_ann[col_ind[i]]))
    return matching_list


def generate_iou_matrix(available_entries_old_ann, unmatched_entries_new_ann, mode):
    """
    generate an matrix filled with the iou value between every bounding box of the old and every bounding box of the
    new annotator
    """
    iou_matrix = np.zeros((len(available_entries_old_ann), len(unmatched_entries_new_ann)))

    for i in range(0, len(available_entries_old_ann)):
        for j in range(0, len(unmatched_entries_new_ann)):
            if mode == 'bbox':
                iou_value = iou.calc_iou_bbox(available_entries_old_ann[i].bbox, unmatched_entries_new_ann[j].bbox)
            if mode == 'seg':
                iou_value = None
                assert "not implemented yet"

            iou_matrix[i][j] = 1-iou_value
    return iou_matrix



