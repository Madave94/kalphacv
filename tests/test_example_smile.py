import pytest
import numpy as np
from kalphacv import calculate_iaa, krippendorff_alpha
from kalphacv import pre_processing
from kalphacv import reliability_data
from kalphacv import hungarian_matching

@pytest.fixture
def annotation_list_smile_fruits():
    file = 'src/smile_fruits.json'
    return calculate_iaa.load_data('coco',file ,'')


@pytest.fixture
def preprocess_smile_fruit(annotation_list_smile_fruits):
    return pre_processing.Preprocess(annotation_list_smile_fruits)


@pytest.fixture
def coincidence_matrix_smile_fruit(preprocess_smile_fruit, iou_threshold=0.5):
    image_name = preprocess_smile_fruit.image_set[0]
    image_annotations = preprocess_smile_fruit.sorted_annotations[image_name]
    image_name_to_images_by_annotator = preprocess_smile_fruit.image_name_to_images_by_annotator[image_name]
    return reliability_data.ReliabilityData(image_name, image_annotations,
                                                image_name_to_images_by_annotator, iou_threshold)


def test_hungarian_matching(coincidence_matrix_smile_fruit, iou_threshold=0.5):
    matching_1_2 = hungarian_matching.run_matching(coincidence_matrix_smile_fruit.all_entries[1],
                                                   coincidence_matrix_smile_fruit.all_entries[2], iou_threshold, 'bbox')
    assert matching_1_2[0][0].id == 1
    assert matching_1_2[0][1].id == 5

    assert matching_1_2[1][0].id == 2
    assert matching_1_2[1][1].id == 6

    assert matching_1_2[2][0].id == 3
    assert matching_1_2[2][1].id == 7

    assert matching_1_2[3][0].id == 4
    assert matching_1_2[3][1].id == 8

    matching_1_3 = hungarian_matching.run_matching(coincidence_matrix_smile_fruit.all_entries[1],
                                                   coincidence_matrix_smile_fruit.all_entries[3], iou_threshold, 'bbox')
    assert matching_1_3[0][0].id == 1
    assert matching_1_3[0][1].id == 9

    assert matching_1_3[1][0].id == 3
    assert matching_1_3[1][1].id == 11

    matching_2_3 = hungarian_matching.run_matching(coincidence_matrix_smile_fruit.all_entries[2],
                                                   coincidence_matrix_smile_fruit.all_entries[3], iou_threshold, 'bbox')
    assert matching_2_3[0][0].id == 5
    assert matching_2_3[0][1].id == 9

    assert matching_2_3[1][0].id == 6
    assert matching_2_3[1][1].id == 10

    assert matching_2_3[2][0].id == 7
    assert matching_2_3[2][1].id == 11

def test_coincidence_matrix(coincidence_matrix_smile_fruit):
    matching_matrix = coincidence_matrix_smile_fruit.run('bbox')

    id_mat = np.zeros((len(matching_matrix), len(matching_matrix[0])))
    cat_mat = np.zeros((len(matching_matrix), len(matching_matrix[0])))

    for row in range(0, len(matching_matrix)):
        for col in range(0, len(matching_matrix[0])):

            id_mat[row][col] = matching_matrix[row][col].id

            if matching_matrix[row][col].category == '*':
                cat_mat[row][col] = np.nan
            else:
                cat_mat[row][col] = matching_matrix[row][col].category

    assert id_mat[0][0] == 1 and id_mat[0][1] == 2 and id_mat[0][2] == 3 and id_mat[0][3] == 4 and id_mat[0][4] == -1
    assert id_mat[1][0] == 5 and id_mat[1][1] == 6 and id_mat[1][2] == 7 and id_mat[1][3] == 8 and id_mat[1][4] == -1
    assert id_mat[2][0] == 9 and id_mat[2][1] == 10 and id_mat[2][2] == 11 and id_mat[2][3] == -1 and id_mat[2][4] == 12

    assert cat_mat[0][0] == 2 and cat_mat[0][1] == 3 and cat_mat[0][2] == 1 and cat_mat[0][3] == 1 and np.isnan(cat_mat[0][4])
    assert cat_mat[1][0] == 2 and cat_mat[1][1] == 3 and cat_mat[1][2] == 1 and cat_mat[1][3] == 1 and np.isnan(cat_mat[1][4])
    assert cat_mat[2][0] == 2 and cat_mat[2][1] == 3 and cat_mat[2][2] == 2 and np.isnan(cat_mat[2][3]) and cat_mat[2][4] == 1


def test_krippendorff(coincidence_matrix_smile_fruit):
    matching_matrix = coincidence_matrix_smile_fruit.run('bbox')
    alpha = krippendorff_alpha.calculate_alpha(matching_matrix)
    assert alpha == 0.494









