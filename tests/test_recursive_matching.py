from test_resources import *
from test_recursive_matching_resources import *

def test_preprocess(annotation_list):
    preprocess = Preprocess(annotation_list)
    assert "images" in preprocess.annotation_list[0]
    assert "annotations" in preprocess.annotation_list[0]
    # all elements should only contain exactly one item since they are either the parent dictonary holding the elements or they just hold the image name
    assert len(preprocess.image_set) == 1
    assert len(preprocess.sorted_annotations) == 1
    assert len(preprocess.image_name_to_images_by_annotator) == 1


def test_init_reliability_matrix(annotation_list):
    preprocess = Preprocess(annotation_list)
    # extract all elements
    image_set = preprocess.image_set
    # run throuhg all unique images that allow calculation of iaa (at least two annotator)
    for image_name in image_set:
        image_annotations = preprocess.sorted_annotations[image_name]
        image_name_to_images_by_annotator = preprocess.image_name_to_images_by_annotator[image_name]
        reliability_data = ReliabilityData(image_name, image_annotations, image_name_to_images_by_annotator,
                                           iou_threshold=0.5)
        # TODO add assert statements


def test_lowest_element_in_set(reliability_matrix):
    lowest_annotator = reliability_matrix.get_lowest_element_in_set({1, 2, 3})
    assert lowest_annotator == 1


def test_matching_randomly(reliability_matrix):
    coincidence_matrix = reliability_matrix.run("bbox")  # this is using the __call__ function
    # should match ME1, ME2 and ME3
    col_1 = [row[0] for row in coincidence_matrix]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    assert col_1[2].id == 3
    # should match ME4 and ME5
    col_2 = [row[1] for row in coincidence_matrix]
    assert col_2[0].id == 4
    assert col_2[1].id == 5
    assert col_2[2].id == -1
    # ME7 should be solo
    col_3 = [row[2] for row in coincidence_matrix]
    assert col_3[0].id == 7
    assert col_3[1].id == -1
    assert col_3[2].id == -1
    # should match ME8 and ME9

    # ME6 should be solo


def rel_matrix(annotation_list):
    preprocess = Preprocess(annotation_list)
    # extract all elements
    image_set = preprocess.image_set
    # run throuhg all unique images that allow calculation of iaa (at least two annotator)
    for image_name in image_set:
        image_annotations = preprocess.sorted_annotations[image_name]
        image_name_to_images_by_annotator = preprocess.image_name_to_images_by_annotator[image_name]
        return ReliabilityData(image_name, image_annotations, image_name_to_images_by_annotator, iou_threshold=0.5)


"""lab testing recursive matching"""


# 1 bbox per annotator
def test_recursive_matching_1_bbox(ann_lst_match, ann_lst_pmatch, ann_lst_nmatch):
    rmat_match = rel_matrix(ann_lst_match)
    rmat_pmatch = rel_matrix(ann_lst_pmatch)
    rmat_nmatch = rel_matrix(ann_lst_nmatch)

    cmat_match = rmat_match.run("bbox")
    cmat_nmatch = rmat_pmatch.run("bbox")
    cmat_pmatch = rmat_nmatch.run("bbox")

    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2

    col = [row[0] for row in cmat_nmatch]
    assert col[0].id == 7
    assert col[1].id == 8

    col = [row[0] for row in cmat_pmatch]
    assert col[0].id == 13
    assert col[1].id == -1
    col_2 = [row[1] for row in cmat_pmatch]
    assert col_2[0].id == -1
    assert col_2[1].id == 14


# 2 bboxes per annotator
def test_rec_2_match(ann_lst_match_match):
    rmat_match = rel_matrix(ann_lst_match_match)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 3
    assert col_2[1].id == 4


def test_rec_2_pmatch(ann_lst_pmatch_pmatch):
    rmat_match = rel_matrix(ann_lst_pmatch_pmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 7
    assert col_1[1].id == 8
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 9
    assert col_2[1].id == 10


def test_rec_2_nmatch(ann_lst_nmatch_nmatch):
    rmat_match = rel_matrix(ann_lst_nmatch_nmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 13
    assert col_1[1].id == -1
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 15
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == 14
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 16

def test_rec_match_pmatch(ann_lst_match_pmatch):
    rmat_match = rel_matrix(ann_lst_match_pmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 9
    assert col_2[1].id == 10

def test_rec_pmatch_nmatch(ann_lst_pmatch_nmatch):
    rmat_match = rel_matrix(ann_lst_pmatch_nmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 7
    assert col_1[1].id == 8
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 15
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == 16

def test_rec_nmatch_match(ann_lst_nmatch_match):
    rmat_match = rel_matrix(ann_lst_nmatch_match)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 3
    assert col_1[1].id == 4
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 13
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == 14

# 3 bboxes per annotator

def test_rec_3_matches(ann_lst_3_matches):
    rmat_match = rel_matrix(ann_lst_3_matches)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 3
    assert col_2[1].id == 4
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 5
    assert col_3[1].id == 6

def test_rec_3_pmatches(ann_lst_3_pmatches):
    rmat_match = rel_matrix(ann_lst_3_pmatches)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 7
    assert col_1[1].id == 8
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 9
    assert col_2[1].id == 10
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 11
    assert col_3[1].id == 12

def test_rec_3_nmatches(ann_lst_3_nmatches):
    rmat_match = rel_matrix(ann_lst_3_nmatches)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 13
    assert col_1[1].id == -1
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 15
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 17
    assert col_3[1].id == -1
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 14
    col_5 = [row[4] for row in cmat_match]
    assert col_5[0].id == -1
    assert col_5[1].id == 16
    col_6 = [row[5] for row in cmat_match]
    assert col_6[0].id == -1
    assert col_6[1].id == 18

def test_rec_match_match_nmatch(ann_lst_match_match_nmatch):
    rmat_match = rel_matrix(ann_lst_match_match_nmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 3
    assert col_2[1].id == 4
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 17
    assert col_3[1].id == -1
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 18

def test_rec_match_pmatch_nmatch(ann_lst_match_pmatch_nmatch):
    rmat_match = rel_matrix(ann_lst_match_pmatch_nmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 9
    assert col_2[1].id == 10
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 17
    assert col_3[1].id == -1
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 18

def test_rec_nmatch_nmatch_pmatch(ann_lst_nmatch_nmatch_pmatch):
    rmat_match = rel_matrix(ann_lst_nmatch_nmatch_pmatch)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 11
    assert col_1[1].id == 12
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 13
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 15
    assert col_3[1].id == -1
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 14
    col_5 = [row[4] for row in cmat_match]
    assert col_5[0].id == -1
    assert col_5[1].id == 16

def test_rec_unbalanced_A_1_B_2(ann_lst_unbalanced_A_1_B_2):
    rmat_match = rel_matrix(ann_lst_unbalanced_A_1_B_2)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 7
    assert col_1[1].id == 8
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == -1
    assert col_2[1].id == 10


def test_rec_unbalanced_A_3_B_1(ann_lst_unbalanced_A_3_B_1):
    rmat_match = rel_matrix(ann_lst_unbalanced_A_3_B_1)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == 2
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 9
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 15
    assert col_3[1].id == -1

def test_rec_unbalanced_A_4_B_1(ann_lst_unbalanced_A_4_B_1):
    rmat_match = rel_matrix(ann_lst_unbalanced_A_4_B_1)
    cmat_match = rmat_match.run("bbox")
    col_1 = [row[0] for row in cmat_match]
    assert col_1[0].id == 1
    assert col_1[1].id == -1
    col_2 = [row[1] for row in cmat_match]
    assert col_2[0].id == 3
    assert col_2[1].id == -1
    col_3 = [row[2] for row in cmat_match]
    assert col_3[0].id == 17
    assert col_3[1].id == -1
    col_4 = [row[3] for row in cmat_match]
    assert col_4[0].id == 18
    assert col_4[1].id == -1
    col_5 = [row[4] for row in cmat_match]
    assert col_5[0].id == -1
    assert col_5[1].id == 19

#3 annptators 1 bbox per annotator
def test_rec_3_anns(ann3_lst_match, ann3_lst_pmatch, ann3_lst_nmatch):
    rmat_match = rel_matrix(ann3_lst_match)
    rmat_pmatch = rel_matrix(ann3_lst_pmatch)
    rmat_nmatch = rel_matrix(ann3_lst_nmatch)

    cmat_match = rmat_match.run("bbox")
    cmat_pmatch = rmat_pmatch.run("bbox")
    cmat_nmatch = rmat_nmatch.run("bbox")

    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2
    assert col[2].id == 20
    col = [row[0] for row in cmat_pmatch]
    assert col[0].id == 7
    assert col[1].id == 8
    assert col[2].id == 23
    col = [row[0] for row in cmat_nmatch]
    assert col[0].id == 13
    assert col[1].id == -1
    assert col[2].id == -1
    col_2 = [row[1] for row in cmat_nmatch]
    assert col_2[0].id == -1
    assert col_2[1].id == 14
    assert col_2[2].id == -1
    col_3 = [row[2] for row in cmat_nmatch]
    assert col_3[0].id == -1
    assert col_3[1].id == -1
    assert col_3[2].id == 26

def test_rec_mmm_mmm(ann3_mmm_mmm):
    rmat_match = rel_matrix(ann3_mmm_mmm)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2
    assert col[2].id == 20
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 3
    assert col_1[1].id == 4
    assert col_1[2].id == 21


def test_rec_ppp_ppp(ann3_ppp_ppp):
    rmat_match = rel_matrix(ann3_ppp_ppp)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 7
    assert col[1].id == 8
    assert col[2].id == 23
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 9
    assert col_1[1].id == 10
    assert col_1[2].id == 24

def test_rec_nnn_nnn(ann3_nnn_nnn):
    rmat_match = rel_matrix(ann3_nnn_nnn)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 13
    assert col[1].id == -1
    assert col[2].id == -1
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 15
    assert col_1[1].id == -1
    assert col_1[2].id == -1
    col_2 = [row[2] for row in cmat_match]
    assert col_2[0].id == -1
    assert col_2[1].id == 14
    assert col_2[2].id == -1
    col_3 = [row[3] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == 16
    assert col_3[2].id == -1
    col_4 = [row[4] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == -1
    assert col_4[2].id == 26
    col_5 = [row[5] for row in cmat_match]
    assert col_5[0].id == -1
    assert col_5[1].id == -1
    assert col_5[2].id == 27


def test_rec_mmp_nmp(ann3_mmp_nmp):
    rmat_match = rel_matrix(ann3_mmp_nmp)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2
    assert col[2].id == 23
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 15
    assert col_1[1].id == -1
    assert col_1[2].id == -1
    col_2 = [row[2] for row in cmat_match]
    assert col_2[0].id == -1
    assert col_2[1].id == 4
    assert col_2[2].id == 24


def test_rec_ppp_pmp(ann3_ppp_pmp):
    rmat_match = rel_matrix(ann3_ppp_pmp)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 7
    assert col[1].id == 8
    assert col[2].id == 23
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 9
    assert col_1[1].id == 4
    assert col_1[2].id == 24


def test_rec_nnn_mpn(ann3_nnn_mpn):
    rmat_match = rel_matrix(ann3_nnn_mpn)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 3
    assert col[1].id == 10
    assert col[2].id == -1
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 13
    assert col_1[1].id == -1
    assert col_1[2].id == -1
    col_2 = [row[2] for row in cmat_match]
    assert col_2[0].id == -1
    assert col_2[1].id == 14
    assert col_2[2].id == -1
    col_3 = [row[3] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == -1
    assert col_3[2].id == 26
    col_4 = [row[4] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == -1
    assert col_4[2].id == 27


def test_rec_unbalanced_A_2_B_1_C_1(ann3_unbalanced_A2_B_1_C_1):
    rmat_match = rel_matrix(ann3_unbalanced_A2_B_1_C_1)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2
    assert col[2].id == 23
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 15
    assert col_1[1].id == -1
    assert col_1[2].id == -1


def test_rec_unbalanced_A_1_B_4_C2(ann3_unbalanced_A_1_B_4_C2):
    rmat_match = rel_matrix(ann3_unbalanced_A_1_B_4_C2)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 7
    assert col[1].id == -1
    assert col[2].id == -1
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == -1
    assert col_1[1].id == 4
    assert col_1[2].id == -1
    col_2 = [row[2] for row in cmat_match]
    assert col_2[0].id == -1
    assert col_2[1].id == 10
    assert col_2[2].id == 24
    col_3 = [row[3] for row in cmat_match]
    assert col_3[0].id == -1
    assert col_3[1].id == 14
    assert col_3[2].id == -1
    col_4 = [row[4] for row in cmat_match]
    assert col_4[0].id == -1
    assert col_4[1].id == 18
    assert col_4[2].id == -1
    col_5 = [row[5] for row in cmat_match]
    assert col_5[0].id == -1
    assert col_5[1].id == -1
    assert col_5[2].id == 26



def test_rec_unbalanced_A_3_B_2_C1(ann3_unbalanced_A_3_B_2_C1):
    rmat_match = rel_matrix(ann3_unbalanced_A_3_B_2_C1)
    cmat_match = rmat_match.run("bbox")
    col = [row[0] for row in cmat_match]
    assert col[0].id == 1
    assert col[1].id == 2
    assert col[2].id == 20
    col_1 = [row[1] for row in cmat_match]
    assert col_1[0].id == 3
    assert col_1[1].id == 10
    assert col_1[2].id == -1
    col_2 = [row[2] for row in cmat_match]
    assert col_2[0].id == 5
    assert col_2[1].id == -1
    assert col_2[2].id == -1