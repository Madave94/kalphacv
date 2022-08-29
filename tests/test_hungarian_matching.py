from test_resources import *

"""------ test run matching --------"""

def test_run_matching_1_box(me_left_match_ann_A, me_left_match_ann_B, me_left_pmatch_ann_A, me_left_pmatch_ann_B,
                            me_left_nmatch_ann_A, me_left_nmatch_ann_B):
    """
    test matching on 1 bbox per annotator
    """
    matching1 = hungarian_matching.run_matching([me_left_match_ann_A], [me_left_match_ann_B], 0.5, 'bbox')
    assert matching1[0][0].id == 1
    assert matching1[0][1].id == 2

    matching2 = hungarian_matching.run_matching([me_left_pmatch_ann_A], [me_left_pmatch_ann_B], 0.5, 'bbox')

    assert matching2[0][0].id == 7
    assert matching2[0][1].id == 8

    matching3 = hungarian_matching.run_matching([me_left_nmatch_ann_A], [me_left_nmatch_ann_B], 0.5, 'bbox')

    assert len(matching3) == 0

# 2 bboxes per annotator
def test_2_matching(me_left_match_ann_A, me_left_match_ann_B, me_center_match_ann_A, me_center_match_ann_B):
    A = [me_left_match_ann_A,me_center_match_ann_A ]
    B = [me_center_match_ann_B, me_left_match_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert matching_matrix[1][0].id == 3
    assert matching_matrix[1][1].id == 4

def test_2_pmatching(me_left_pmatch_ann_A, me_left_pmatch_ann_B, me_center_pmatch_ann_A, me_center_pmatch_ann_B):
    A = [me_left_pmatch_ann_A, me_center_pmatch_ann_A]
    B = [me_left_pmatch_ann_B, me_center_pmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 7
    assert matching_matrix[0][1].id == 8
    assert matching_matrix[1][0].id == 9
    assert matching_matrix[1][1].id == 10

def test_2_nmatching(me_left_nmatch_ann_A, me_left_nmatch_ann_B, me_center_nmatch_ann_A, me_center_nmatch_ann_B):
    A = [me_left_nmatch_ann_A, me_center_nmatch_ann_A]
    B = [me_left_nmatch_ann_B,me_center_nmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert len(matching_matrix) == 0


def test_match_pmatch(me_left_match_ann_A, me_left_match_ann_B, me_center_pmatch_ann_A, me_center_pmatch_ann_B):
    A = [me_left_match_ann_A,me_center_pmatch_ann_A]
    B = [me_center_pmatch_ann_B,me_left_match_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert matching_matrix[1][0].id == 9
    assert matching_matrix[1][1].id == 10

def test_pmatch_nmatch(me_left_pmatch_ann_A, me_left_pmatch_ann_B, me_center_nmatch_ann_A, me_center_nmatch_ann_B):
    A = [me_left_pmatch_ann_A,me_center_nmatch_ann_A]
    B = [me_left_pmatch_ann_B,me_center_nmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 7
    assert matching_matrix[0][1].id == 8

def test_nmatch_match(me_left_nmatch_ann_A, me_left_nmatch_ann_B, me_center_match_ann_A, me_center_match_ann_B):
    A = [me_center_match_ann_A,me_left_nmatch_ann_A]
    B = [me_left_nmatch_ann_B,me_center_match_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 3
    assert matching_matrix[0][1].id == 4

# trivial cases 3 bboxes per annotator
def test_3_matches(me_left_match_ann_A, me_left_match_ann_B, me_center_match_ann_A, me_center_match_ann_B, me_right_match_ann_A, me_right_match_ann_B):
    A = [me_left_match_ann_A,me_center_match_ann_A, me_right_match_ann_A]
    B = [me_left_match_ann_B,me_center_match_ann_B, me_right_match_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert matching_matrix[1][0].id == 3
    assert matching_matrix[1][1].id == 4
    assert matching_matrix[2][0].id == 5
    assert matching_matrix[2][1].id == 6

def test_3_pmatches(me_left_pmatch_ann_A, me_left_pmatch_ann_B, me_center_pmatch_ann_A, me_center_pmatch_ann_B, me_right_pmatch_ann_A, me_right_pmatch_ann_B):
    A = [me_left_pmatch_ann_A,me_center_pmatch_ann_A,me_right_pmatch_ann_A]
    B = [me_center_pmatch_ann_B,me_right_pmatch_ann_B,me_left_pmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 7
    assert matching_matrix[0][1].id == 8
    assert matching_matrix[1][0].id == 9
    assert matching_matrix[1][1].id == 10
    assert matching_matrix[2][0].id == 11
    assert matching_matrix[2][1].id == 12

def test_3_nmatches(me_left_nmatch_ann_A, me_left_nmatch_ann_B, me_center_nmatch_ann_A, me_center_nmatch_ann_B, me_right_nmatch_ann_A, me_right_nmatch_ann_B):
    A = [me_center_nmatch_ann_A,me_left_nmatch_ann_A,me_right_nmatch_ann_A]
    B = [me_left_nmatch_ann_B,me_center_nmatch_ann_B,me_right_nmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert len(matching_matrix) == 0

def test_match_match_nmatch(me_left_match_ann_A, me_left_match_ann_B, me_center_match_ann_A, me_center_match_ann_B, me_right_nmatch_ann_A, me_right_nmatch_ann_B):
    A = [me_left_match_ann_A,me_center_match_ann_A,me_right_nmatch_ann_A]
    B = [me_left_match_ann_B,me_center_match_ann_B,me_right_nmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert matching_matrix[1][0].id == 3
    assert matching_matrix[1][1].id == 4
    assert len(matching_matrix) == 2

def test_match_pmatch_nmatch(me_left_match_ann_A, me_left_match_ann_B,me_center_nmatch_ann_A, me_center_nmatch_ann_B,me_right_pmatch_ann_A, me_right_pmatch_ann_B):
    A = [me_left_match_ann_A,me_center_nmatch_ann_A,me_right_pmatch_ann_A]
    B = [me_left_match_ann_B,me_center_nmatch_ann_B,me_right_pmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert matching_matrix[1][0].id == 11
    assert matching_matrix[1][1].id == 12
    assert len(matching_matrix) == 2

def test_nmatch_nmatch_pmatch(me_left_nmatch_ann_A, me_left_nmatch_ann_B, me_center_nmatch_ann_A, me_center_nmatch_ann_B,me_right_pmatch_ann_A, me_right_pmatch_ann_B):
    A = [me_left_nmatch_ann_A,me_center_nmatch_ann_A,me_right_pmatch_ann_A]
    B = [me_left_nmatch_ann_B,me_center_nmatch_ann_B,me_right_pmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 11
    assert matching_matrix[0][1].id == 12
    assert len(matching_matrix) == 1

# edge cases
def test_unbalanced_A_1_B_2(me_left_pmatch_ann_A, me_left_pmatch_ann_B, me_center_pmatch_ann_B):
    A = [me_left_pmatch_ann_A]
    B = [me_left_pmatch_ann_B,me_center_pmatch_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 7
    assert matching_matrix[0][1].id == 8
    assert len(matching_matrix) == 1

def test_unbalanced_A_3_B_1(me_left_match_ann_A, me_left_match_ann_B, me_center_pmatch_ann_A, me_center_pmatch_ann_B):
    A = [me_left_match_ann_A,me_center_pmatch_ann_A, me_center_pmatch_ann_B]
    B = [me_left_match_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert matching_matrix[0][0].id == 1
    assert matching_matrix[0][1].id == 2
    assert len(matching_matrix) == 1

def test_unbalanced_A_4_B_1(me_left_match_ann_A, me_center_match_ann_A, me_right_nmatch_ann_A, me_right_nmatch_ann_B, me_large_bbox_ann_B):
    A = [me_left_match_ann_A,me_center_match_ann_A, me_right_nmatch_ann_A, me_right_nmatch_ann_B]
    B = [me_large_bbox_ann_B]

    matching_matrix = hungarian_matching.run_matching(A,B,0.5,'bbox')
    assert len(matching_matrix) == 0


# 3 Annotators, 1 bbox per annotator




