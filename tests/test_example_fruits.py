import pytest
import numpy as np
from iaadet import calculate_iaa, krippendorff_alpha
from iaadet import pre_processing
from iaadet import reliability_data
from iaadet import hungarian_matching
from iaadet import iou

@pytest.fixture
def annotation_list_fruits():
    file = 'src/fruits.json'
    return calculate_iaa.load_data('coco',file ,'')


@pytest.fixture
def preprocess_fruit(annotation_list_fruits):
    return pre_processing.Preprocess(annotation_list_fruits)


def test_preprocess(preprocess_fruit):
    assert len(preprocess_fruit.sorted_annotations['apple_banana.jpg']) == 9
    assert len(preprocess_fruit.category_list) == 2
    assert len(preprocess_fruit.image_name_to_images_by_annotator['apple_banana.jpg']) == 3

@pytest.fixture
def coincidence_matrix_fruit(preprocess_fruit, iou_threshold=0.5):
    image_name = preprocess_fruit.image_set[0]
    image_annotations = preprocess_fruit.sorted_annotations[image_name]
    image_name_to_images_by_annotator = preprocess_fruit.image_name_to_images_by_annotator[image_name]
    return reliability_data.ReliabilityData(image_name, image_annotations,
                                                image_name_to_images_by_annotator, iou_threshold)


def test_iou(coincidence_matrix_fruit):
    assert round(iou.calc_iou_bbox(coincidence_matrix_fruit.all_entries[1][0].bbox, coincidence_matrix_fruit.all_entries[2][2].bbox), 4) == 0.84 #ME1, ME6
    assert round(iou.calc_iou_bbox(coincidence_matrix_fruit.all_entries[1][0].bbox, coincidence_matrix_fruit.all_entries[3][2].bbox), 4) == 0.8808 #ME1, ME9
    assert round(iou.calc_iou_bbox(coincidence_matrix_fruit.all_entries[1][1].bbox, coincidence_matrix_fruit.all_entries[2][1].bbox), 4) == 0.9689 #ME2, ME5
    assert round(iou.calc_iou_bbox(coincidence_matrix_fruit.all_entries[1][2].bbox, coincidence_matrix_fruit.all_entries[2][0].bbox), 4) == 0.8388 #ME3, ME4



def test_hungarian_matching(coincidence_matrix_fruit, iou_threshold=0.5):
    matching_1_2= hungarian_matching.run_matching(coincidence_matrix_fruit.all_entries[1], coincidence_matrix_fruit.all_entries[2], iou_threshold, 'bbox')
    assert matching_1_2[0][0].id == 1
    assert matching_1_2[0][1].id == 6

    assert matching_1_2[1][0].id == 2
    assert matching_1_2[1][1].id == 5

    assert matching_1_2[2][0].id == 3
    assert matching_1_2[2][1].id == 4

    matching_1_3= hungarian_matching.run_matching(coincidence_matrix_fruit.all_entries[1], coincidence_matrix_fruit.all_entries[3], iou_threshold, 'bbox')
    assert matching_1_3[0][0].id == 1
    assert matching_1_3[0][1].id == 9

    assert matching_1_3[1][0].id == 3
    assert matching_1_3[1][1].id == 8

    matching_2_3 = hungarian_matching.run_matching(coincidence_matrix_fruit.all_entries[2], coincidence_matrix_fruit.all_entries[3], iou_threshold, 'bbox')
    assert matching_2_3[0][0].id == 4
    assert matching_2_3[0][1].id == 8

    assert matching_2_3[1][0].id == 6
    assert matching_2_3[1][1].id == 9


def test_coincidence_matrix(coincidence_matrix_fruit):
    matching_matrix = coincidence_matrix_fruit.run('bbox')

    id_mat = np.zeros((len(matching_matrix), len(matching_matrix[0])))
    cat_mat = np.zeros((len(matching_matrix), len(matching_matrix[0])))

    for row in range(0, len(matching_matrix)):
        for col in range(0, len(matching_matrix[0])):

            id_mat[row][col] = matching_matrix[row][col].id

            if matching_matrix[row][col].category == '*':
                cat_mat[row][col] = np.nan
            else:
                cat_mat[row][col] = matching_matrix[row][col].category

    assert id_mat[0][0] == 1 and id_mat[0][1] == 2 and id_mat[0][2] == 3 and id_mat[0][3] == -1
    assert id_mat[1][0] == 6 and id_mat[1][1] == 5 and id_mat[1][2] == 4 and id_mat[1][3] == -1
    assert id_mat[2][0] == 9 and id_mat[2][1] == -1 and id_mat[2][2] == 8 and id_mat[2][3] == 7

    assert cat_mat[0][0] == 2 and cat_mat[0][1] == 1 and cat_mat[0][2] == 1 and np.isnan(cat_mat[0][3])
    assert cat_mat[1][0] == 2 and cat_mat[1][1] == 1 and cat_mat[1][2] == 1 and np.isnan(cat_mat[1][3])
    assert cat_mat[2][0] == 1 and np.isnan(cat_mat[2][1]) and cat_mat[2][2] == 2 and cat_mat[2][3] == 1

def test_krippendorff(coincidence_matrix_fruit):
    matching_matrix = coincidence_matrix_fruit.run('bbox')
    alpha = krippendorff_alpha.calculate_alpha(matching_matrix)
    assert alpha == 0.0222

@pytest.fixture
def matching_matrix_fruits(coincidence_matrix_fruit):
    return coincidence_matrix_fruit.run('bbox')