import pytest
import numpy as np
from kalphacv import calculate_iaa, krippendorff_alpha
from kalphacv import pre_processing
from kalphacv import reliability_data

@pytest.fixture
def annotation_list_landscape():
    file = 'src/landscape_annotations.json'
    return calculate_iaa.load_data('coco',file ,'')

@pytest.fixture
def preprocess(annotation_list_landscape):
    return pre_processing.Preprocess(annotation_list_landscape)


@pytest.fixture
def coincidence_matrix(preprocess, iou_threshold=0.5):
    image_name = preprocess.image_set[0]
    image_annotations = preprocess.sorted_annotations[image_name]
    image_name_to_images_by_annotator = preprocess.image_name_to_images_by_annotator[image_name]
    rel_data = reliability_data.ReliabilityData(image_name, image_annotations,
                                                image_name_to_images_by_annotator, iou_threshold)
    return rel_data.run('bbox')


def test_coincidence_matrix(coincidence_matrix):

    id_mat = np.zeros((len(coincidence_matrix), len(coincidence_matrix[0])))
    for row in range(0, len(coincidence_matrix)):
        for col in range(0, len(coincidence_matrix[0])):
            id_mat[row][col] = coincidence_matrix[row][col].id
    assert id_mat[0][0] == 1 and id_mat[0][1] == 2 and id_mat[0][2] == 3
    assert id_mat[0][3] == 4 and id_mat[0][4] == 5 and id_mat[0][5] == 6
    assert id_mat[0][6] == 7 and id_mat[0][7] == 8 and id_mat[0][8] == 9
    assert id_mat[0][9] == -1 and id_mat[0][10] == -1 and id_mat[0][11] == -1
    assert id_mat[0][12] == -1 and id_mat[0][13] == -1

    assert id_mat[1][0] == 15 and id_mat[1][1] == 11 and id_mat[1][2] == 16 and id_mat[1][3] == -1
    assert id_mat[1][4] == -1 and id_mat[1][5] == 13 and id_mat[1][6] == 14 and id_mat[1][7] == 10
    assert id_mat[1][8] == 19 and id_mat[1][9] == 12 and id_mat[1][10] == 17 and id_mat[1][11] == 18
    assert id_mat[1][12] == -1 and id_mat[1][13] == -1

    assert id_mat[2][0] == 23 and id_mat[2][1] == 30 and id_mat[2][2] == -1 and id_mat[2][3] == 29
    assert id_mat[2][4] == 22 and id_mat[2][5] == 21 and id_mat[2][6] == 26 and id_mat[2][7] == 27
    assert id_mat[2][8] == -1 and id_mat[2][9] == -1 and id_mat[2][10] == 25 and id_mat[2][11] == 20
    assert id_mat[2][12] == 24 and id_mat[2][13] == 28

def test_krippendorff(coincidence_matrix):
    alpha = krippendorff_alpha.calculate_alpha(coincidence_matrix)
    assert alpha == 0.4889
