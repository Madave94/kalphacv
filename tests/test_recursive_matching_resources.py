import pytest

"""-------------test recursive matching-------------"""
# 1 box per annotator
@pytest.fixture
def ann_lst_match(combine_images, left_match_ann_A, left_match_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B],
    }]

@pytest.fixture
def ann_lst_pmatch(combine_images, left_pmatch_ann_A, left_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B],
    }]

@pytest.fixture
def ann_lst_nmatch(combine_images, left_nmatch_ann_A, left_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B],
    }]

# 2 boxes per annotator

@pytest.fixture
def ann_lst_match_match(combine_images, left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B],
    }]


@pytest.fixture
def ann_lst_pmatch_pmatch(combine_images, left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_A, center_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_A, center_pmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_nmatch_nmatch(combine_images, left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_match_pmatch(combine_images, left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_pmatch_ann_B],
    }]

@pytest.fixture
def ann_lst_pmatch_nmatch(combine_images, left_pmatch_ann_A, left_pmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_nmatch_match(combine_images, left_nmatch_ann_A, left_nmatch_ann_B, center_match_ann_A, center_match_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, center_match_ann_A, center_match_ann_B],
    }]

# 3 boxes per annotator


@pytest.fixture
def ann_lst_3_matches(combine_images, left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B, right_match_ann_A, right_match_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B, right_match_ann_A, right_match_ann_B],
    }]


@pytest.fixture
def ann_lst_3_pmatches(combine_images, left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_A, center_pmatch_ann_B, right_pmatch_ann_A, right_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_A, center_pmatch_ann_B, right_pmatch_ann_A, right_pmatch_ann_B],
    }]

@pytest.fixture
def ann_lst_3_nmatches(combine_images, left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B, right_nmatch_ann_A, right_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B, right_nmatch_ann_A, right_nmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_match_match_nmatch(combine_images, left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B, right_nmatch_ann_A, right_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_match_ann_A, center_match_ann_B, right_nmatch_ann_A, right_nmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_match_pmatch_nmatch(combine_images, left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_pmatch_ann_B, right_nmatch_ann_A, right_nmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_pmatch_ann_B, right_nmatch_ann_A, right_nmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_nmatch_nmatch_pmatch(combine_images, left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B, right_pmatch_ann_A, right_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, center_nmatch_ann_A, center_nmatch_ann_B, right_pmatch_ann_A, right_pmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_unbalanced_A_1_B_2(combine_images, left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_B):
    return [{
        "images": combine_images,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, center_pmatch_ann_B],
    }]


@pytest.fixture
def ann_lst_unbalanced_A_3_B_1(combine_images, left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_nmatch_ann_A):
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, left_match_ann_B, center_pmatch_ann_A, center_nmatch_ann_A],
    }]

@pytest.fixture
def ann_lst_unbalanced_A_4_B_1(combine_images, left_match_ann_A, center_match_ann_A, right_nmatch_ann_A, right_nmatch_ann_B, large_box):
    right_nmatch_ann_B['image_id'] = 1
    return [{
        "images": combine_images,
        "annotations": [left_match_ann_A, center_match_ann_A, right_nmatch_ann_A, right_nmatch_ann_B, large_box],
    }]

# 3 Annotators
@pytest.fixture
def ann3_lst_match(combine_images3, left_match_ann_A, left_match_ann_B, left_match_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_match_ann_A, left_match_ann_B, left_match_ann_C],
    }]

@pytest.fixture
def ann3_lst_pmatch(combine_images3, left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C):

    return [{
        "images": combine_images3,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C],
    }]

@pytest.fixture
def ann3_lst_nmatch(combine_images3, left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C],
    }]

# 3 Annotators, 2 bboxes per annotator
# m = match
# p = partial match/overlap
# n = no match
# x = no entry
@pytest.fixture
def ann3_mmm_mmm(combine_images3, left_match_ann_A, left_match_ann_B, left_match_ann_C,  center_match_ann_A, center_match_ann_B, center_match_ann_C):

    return [{
        "images": combine_images3,
        "annotations": [left_match_ann_A, left_match_ann_B, left_match_ann_C,  center_match_ann_A, center_match_ann_B, center_match_ann_C ],
    }]

@pytest.fixture
def ann3_ppp_ppp(combine_images3, left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C, center_pmatch_ann_A, center_pmatch_ann_B, center_pmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C, center_pmatch_ann_A, center_pmatch_ann_B, center_pmatch_ann_C],
    }]

@pytest.fixture
def ann3_nnn_nnn(combine_images3, left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_nmatch_ann_A, center_nmatch_ann_B, center_nmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_nmatch_ann_A, center_nmatch_ann_B, center_nmatch_ann_C],
    }]


@pytest.fixture
def ann3_mmp_nmp(combine_images3, left_match_ann_A, left_match_ann_B, left_pmatch_ann_C, center_nmatch_ann_A, center_match_ann_B, center_pmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_match_ann_A, left_match_ann_B, left_pmatch_ann_C, center_nmatch_ann_A, center_match_ann_B, center_pmatch_ann_C],
    }]

@pytest.fixture
def ann3_ppp_pmp(combine_images3, left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C, center_pmatch_ann_A, center_match_ann_B, center_pmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_pmatch_ann_A, left_pmatch_ann_B, left_pmatch_ann_C, center_pmatch_ann_A, center_match_ann_B, center_pmatch_ann_C],
    }]

@pytest.fixture
def ann3_nnn_mpn(combine_images3, left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_match_ann_A, center_pmatch_ann_B, center_nmatch_ann_C):
    return [{
        "images": combine_images3,
        "annotations": [left_nmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_match_ann_A, center_pmatch_ann_B, center_nmatch_ann_C],
    }]


@pytest.fixture
def ann3_unbalanced_A2_B_1_C_1(combine_images3, left_match_ann_A, left_match_ann_B, left_pmatch_ann_C, center_nmatch_ann_A):
    return [{
        "images": combine_images3,
        "annotations": [left_match_ann_A, left_match_ann_B, left_pmatch_ann_C, center_nmatch_ann_A],
    }]

@pytest.fixture
def ann3_unbalanced_A_1_B_4_C2(combine_images3, left_pmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_pmatch_ann_B, center_match_ann_B, center_pmatch_ann_C, right_nmatch_ann_B):
    return [{
        "images": combine_images3,
        "annotations": [left_pmatch_ann_A, left_nmatch_ann_B, left_nmatch_ann_C, center_pmatch_ann_B, center_match_ann_B, center_pmatch_ann_C, right_nmatch_ann_B],
    }]

@pytest.fixture
def ann3_unbalanced_A_3_B_2_C1(combine_images3, left_match_ann_A, left_match_ann_B, left_match_ann_C, center_match_ann_A, center_pmatch_ann_B, right_match_ann_A):
    return [{
        "images": combine_images3,
        "annotations": [left_match_ann_A, left_match_ann_B, left_match_ann_C, center_match_ann_A, center_pmatch_ann_B, right_match_ann_A],
    }]
