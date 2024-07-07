import pytest
from kalphacv.pre_processing import *
from kalphacv.reliability_data import *


"""-------- Annotators ------------"""


@pytest.fixture
def test_image_ann_A():
    return {"id": 1, "file_name": "test.jpg", "annotator": "ann_A"}


@pytest.fixture
def test_image_ann_B():
    return {"id": 2, "file_name": "test.jpg", "annotator": "ann_B"}


@pytest.fixture
def test_image_ann_C():
    return {"id": 3, "file_name": "test.jpg", "annotator": "ann_C"}


"""-------- Annotations for recursive matching ------------"""


@pytest.fixture
def entry_1_ann_A():
    return {"bbox": [0, 0, 3, 4], "category_id": 1, "image_id": 1,
            'id': 1, 'attributes': {"media type": "alpha", "content type": "alpha", "ID": "alpha"},
            'segmentation': [0, 0, 3, 4]}


@pytest.fixture
def entry_1_ann_B():
    return {"bbox": [0, 0, 3, 4], "category_id": 1, "image_id": 2,
            'id': 2, 'attributes': {"media type": "alpha", "content type": "alpha", "ID": "alpha"},
            'segmentation': [0, 0, 3, 4]}


@pytest.fixture
def entry_1_ann_C():
    return {"bbox": [0, 0, 3, 4], "category_id": 1, "image_id": 3,
            'id': 3, 'attributes': {"media type": "alpha", "content type": "alpha", "ID": "alpha"},
            'segmentation': [0, 0, 3, 4]}


@pytest.fixture
def entry_2_ann_A():
    return {"bbox": [5, 5, 3, 4], "category_id": 2, "image_id": 1,
            'id': 4, 'attributes': {"media type": "gamma", "content type": "gamma", "ID": "gamma"},
            'segmentation': [5, 5, 3, 4]}


@pytest.fixture
def entry_2_ann_B():
    return {"bbox": [5, 5, 3, 4], "category_id": 2, "image_id": 2,
            'id': 5, 'attributes': {"media type": "gamma", "content type": "gamma", "ID": "gamma"},
            'segmentation': [5, 5, 3, 4]}


@pytest.fixture
def entry_2_ann_C():
    return {"bbox": [8, 7, 2, 6], "category_id": 2, "image_id": 3,
            'id': 6, 'attributes': {}, 'segmentation': [8, 7, 2, 6]}


@pytest.fixture
def entry_3_ann_A():
    return {"bbox": [14, 14, 4, 4], "category_id": 3, "image_id": 1,
            'id': 7, 'attributes': {}, 'segmentation': [14, 14, 4, 4]}


@pytest.fixture
def entry_3_ann_B():
    return {"bbox": [10, 10, 3, 4], "category_id": 3, "image_id": 2,
            'id': 8, 'attributes': {"media type": "beta", "content type": "beta", "ID": "beta"},
            'segmentation': [10, 10, 3, 4]}


@pytest.fixture
def entry_3_ann_C():
    return {"bbox": [10, 10, 3, 4], "category_id": 3, "image_id": 3,
            'id': 9, 'attributes': {"media type": "delta", "content type": "delta", "ID": "delta"},
            'segmentation': [10, 10, 3, 4]}

@pytest.fixture
def combine_images(test_image_ann_A, test_image_ann_B):
    return [test_image_ann_A, test_image_ann_B]

@pytest.fixture
def combine_images3(test_image_ann_A, test_image_ann_B, test_image_ann_C):
    return [test_image_ann_A, test_image_ann_B, test_image_ann_C]


@pytest.fixture
def combine_entry_1(entry_1_ann_A, entry_1_ann_B, entry_1_ann_C):
    return [entry_1_ann_A, entry_1_ann_B, entry_1_ann_C]


@pytest.fixture
def combine_entry_2(entry_2_ann_A, entry_2_ann_B, entry_2_ann_C):
    return [entry_2_ann_A, entry_2_ann_B, entry_2_ann_C]


@pytest.fixture
def combine_entry_3(entry_3_ann_A, entry_3_ann_B, entry_3_ann_C):
    return [entry_3_ann_A, entry_3_ann_B, entry_3_ann_C]


@pytest.fixture
def annotation_list(combine_images3, combine_entry_1, combine_entry_2, combine_entry_3):
    return [{
        "images": combine_images3,
        "annotations": combine_entry_1 + combine_entry_2 + combine_entry_3,
    }]


@pytest.fixture
def reliability_matrix(annotation_list):
    preprocess = Preprocess(annotation_list)
    # extract all elements
    image_set = preprocess.image_set
    # run through all unique images that allow calculation of iaa (at least two annotator)
    for image_name in image_set:
        image_annotations = preprocess.sorted_annotations[image_name]
        image_name_to_images_by_annotator = preprocess.image_name_to_images_by_annotator[image_name]
        return ReliabilityData(image_name, image_annotations, image_name_to_images_by_annotator, iou_threshold=0.5)


"""-------- Annotations for hungarian matching tests ------------"""


# matching
@pytest.fixture
def left_match_ann_A():
    return {"bbox": [0, 0, 10, 10], "category_id": 1, "image_id": 1, 'id': 1, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def left_match_ann_B():
    return {"bbox": [0, 0, 10, 10], "category_id": 1, "image_id": 2, 'id': 2, 'attributes':
        {"ID": "beta"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def center_match_ann_A():
    return {"bbox": [40, 0, 10, 10], "category_id": 2, "image_id": 1, 'id': 3, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 10, 10]}


@pytest.fixture
def center_match_ann_B():
    return {"bbox": [40, 0, 10, 10], "category_id": 2, "image_id": 2, 'id': 4, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "beta"}, 'segmentation': [40, 40, 10, 10]}


@pytest.fixture
def right_match_ann_A():
    return {"bbox": [100, 0, 20, 8], "category_id": 3, "image_id": 1, 'id': 5, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 20, 8]}


@pytest.fixture
def right_match_ann_B():
    return {"bbox": [100, 0, 20, 8], "category_id": 3, "image_id": 2, 'id': 6, 'attributes':
        {"media type": "beta", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 20, 8]}


# partial matching
@pytest.fixture
def left_pmatch_ann_A():
    return {"bbox": [0, 0, 10, 10], "category_id": 1, "image_id": 1, 'id': 7, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def left_pmatch_ann_B():
    return {"bbox": [1, 0, 9, 11], "category_id": 2, "image_id": 2, 'id': 8, 'attributes':
        {"media type": "beta", "content type": "alpha", "ID": "alpha"}, 'segmentation': [1, 0, 9, 11]}


@pytest.fixture
def center_pmatch_ann_A():
    return {"bbox": [40, 0, 10, 15], "category_id": 2, "image_id": 1, 'id': 9, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 10, 15]}


@pytest.fixture
def center_pmatch_ann_B():
    return {"bbox": [41, 0, 9, 16], "category_id": 2, "image_id": 2, 'id': 10, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [41, 40, 9, 16]}


@pytest.fixture
def right_pmatch_ann_A():
    return {"bbox": [100, 1, 10, 25], "category_id": 3, "image_id": 1, 'id': 11, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 10, 25]}


@pytest.fixture
def right_pmatch_ann_B():
    return {"bbox": [100, 0, 11, 22], "category_id": 3, "image_id": 2, 'id': 12, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 99, 11, 22]}


# no matching
@pytest.fixture
def left_nmatch_ann_A():
    return {"bbox": [0, 0, 10, 10], "category_id": 1, "image_id": 1, 'id': 13, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def left_nmatch_ann_B():
    return {"bbox": [11, 1, 10, 10], "category_id": 2, "image_id": 2, 'id': 14, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [11, 11, 10, 10]}


@pytest.fixture
def center_nmatch_ann_A():
    return {"bbox": [40, 0, 5, 5], "category_id": 3, "image_id": 1, 'id': 15, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 5, 5]}


@pytest.fixture
def center_nmatch_ann_B():
    return {"bbox": [50, 0, 7, 9], "category_id": 4, "image_id": 2, 'id': 16, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [50, 50, 7, 9]}


@pytest.fixture
def right_nmatch_ann_A():
    return {"bbox": [100, 0, 33, 21], "category_id": 5, "image_id": 1, 'id': 17, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 33, 21]}


@pytest.fixture
def right_nmatch_ann_B():
    return {"bbox": [130, 0, 50, 66], "category_id": 6, "image_id": 2, 'id': 18, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [130, 110, 50, 66]}


# unbalanced matching
@pytest.fixture
def large_box():
    return {"bbox": [0, 0, 200, 300], "category_id": 7, "image_id": 2, 'id': 19, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 200, 300]}


# 3rd annotator
@pytest.fixture
def left_match_ann_C():
    return {"bbox": [0, 0, 10, 10], "category_id": 1, "image_id": 3, 'id': 20, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def center_match_ann_C():
    return {"bbox": [40, 0, 10, 10], "category_id": 2, "image_id": 3, 'id': 21, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 10, 10]}

@pytest.fixture
def right_match_ann_C():
    return {"bbox": [100, 0, 20, 8], "category_id": 3, "image_id": 3, 'id': 22, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 20, 8]}


@pytest.fixture
def left_pmatch_ann_C():
    return {"bbox": [0, 1, 10, 9], "category_id": 1, "image_id": 3, 'id': 23, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def center_pmatch_ann_C():
    return {"bbox": [41, 0, 10, 15], "category_id": 2, "image_id": 3, 'id': 24, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 10, 15]}

@pytest.fixture
def right_pmatch_ann_C():
    return {"bbox": [100, 1, 10, 23], "category_id": 3, "image_id": 3, 'id': 25, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 10, 25]}

# no match
@pytest.fixture
def left_nmatch_ann_C():
    return {"bbox": [0, 11, 10, 10], "category_id": 1, "image_id": 3, 'id': 26, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [0, 0, 10, 10]}


@pytest.fixture
def center_nmatch_ann_C():
    return {"bbox": [0, 40, 5, 5], "category_id": 3, "image_id": 3, 'id': 27, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [40, 40, 5, 5]}

@pytest.fixture
def right_nmatch_ann_C():
    return {"bbox": [0, 100, 33, 21], "category_id": 5, "image_id": 3, 'id': 28, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 33, 21]}

@pytest.fixture
def category_7():
    return {"bbox": [0, 0, 0, 0], "category_id": 7, "image_id": 3, 'id': 29, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 33, 21]}

@pytest.fixture
def category_8():
    return {"bbox": [0, 100, 33, 21], "category_id": 8, "image_id": 3, 'id': 30, 'attributes':
        {"media type": "alpha", "content type": "alpha", "ID": "alpha"}, 'segmentation': [100, 100, 33, 21]}



"""------ convert to matrix entry"""


@pytest.fixture
def me_left_match_ann_A(left_match_ann_A):
    return reliability_data.MatrixEntry(left_match_ann_A, "ann_A")


@pytest.fixture
def me_left_match_ann_B(left_match_ann_B):
    return reliability_data.MatrixEntry(left_match_ann_B, "ann_B")


@pytest.fixture
def me_center_match_ann_A(center_match_ann_A):
    return reliability_data.MatrixEntry(center_match_ann_A, "ann_A")


@pytest.fixture
def me_center_match_ann_B(center_match_ann_B):
    return reliability_data.MatrixEntry(center_match_ann_B, "ann_B")


@pytest.fixture
def me_right_match_ann_A(right_match_ann_A):
    return reliability_data.MatrixEntry(right_match_ann_A, "ann_A")


@pytest.fixture
def me_right_match_ann_B(right_match_ann_B):
    return reliability_data.MatrixEntry(right_match_ann_B, "ann_B")


# partial matching
@pytest.fixture
def me_left_pmatch_ann_A(left_pmatch_ann_A):
    return reliability_data.MatrixEntry(left_pmatch_ann_A, "ann_A")


@pytest.fixture
def me_left_pmatch_ann_B(left_pmatch_ann_B):
    return reliability_data.MatrixEntry(left_pmatch_ann_B, "ann_B")


@pytest.fixture
def me_center_pmatch_ann_A(center_pmatch_ann_A):
    return reliability_data.MatrixEntry(center_pmatch_ann_A, "ann_A")


@pytest.fixture
def me_center_pmatch_ann_B(center_pmatch_ann_B):
    return reliability_data.MatrixEntry(center_pmatch_ann_B, "ann_B")


@pytest.fixture
def me_right_pmatch_ann_A(right_pmatch_ann_A):
    return reliability_data.MatrixEntry(right_pmatch_ann_A, "ann_A")


@pytest.fixture
def me_right_pmatch_ann_B(right_pmatch_ann_B):
    return reliability_data.MatrixEntry(right_pmatch_ann_B, "ann_B")


# no matching
@pytest.fixture
def me_left_nmatch_ann_A(left_nmatch_ann_A):
    return reliability_data.MatrixEntry(left_nmatch_ann_A, "ann_A")


@pytest.fixture
def me_left_nmatch_ann_B(left_nmatch_ann_B):
    return reliability_data.MatrixEntry(left_nmatch_ann_B, "ann_B")



@pytest.fixture
def me_center_nmatch_ann_A(center_nmatch_ann_A):
    return reliability_data.MatrixEntry(center_nmatch_ann_A, "ann_A")


@pytest.fixture
def me_center_nmatch_ann_B(center_nmatch_ann_B):
    return reliability_data.MatrixEntry(center_nmatch_ann_B, "ann_B")


@pytest.fixture
def me_right_nmatch_ann_A(right_nmatch_ann_A):
    return reliability_data.MatrixEntry(right_nmatch_ann_A, "ann_A")


@pytest.fixture
def me_right_nmatch_ann_B(right_nmatch_ann_B):
    return reliability_data.MatrixEntry(right_nmatch_ann_B, "ann_B")

@pytest.fixture
def me_large_bbox_ann_B(large_box):
    return reliability_data.MatrixEntry(large_box, "ann_B")


@pytest.fixture
def empty_entry_A():
    return reliability_data.EmptyEntry("ann_A")

@pytest.fixture
def empty_entry_B():
    return reliability_data.EmptyEntry("ann_B")

@pytest.fixture
def empty_entry_C():
    return reliability_data.EmptyEntry("ann_C")

@pytest.fixture
def me_category7_ann_B(category_7):
    return reliability_data.MatrixEntry(category_7, "Ann_B")

@pytest.fixture
def me_category8_ann_A(category_8):
    return reliability_data.MatrixEntry(category_8, "Ann_A")