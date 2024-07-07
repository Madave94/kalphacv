from kalphacv import krippendorff_alpha
from test_resources import *

def test_krippendorff(me_left_match_ann_A,me_center_match_ann_A, me_right_match_ann_A, me_right_nmatch_ann_A, me_right_nmatch_ann_B, me_category7_ann_B,me_category8_ann_A, empty_entry_A, empty_entry_B, empty_entry_C):
    cat1 = me_left_match_ann_A
    cat2 = me_center_match_ann_A
    cat3 = me_right_match_ann_A
    cat5 = me_right_nmatch_ann_A
    cat6 = me_right_nmatch_ann_B
    cat7 = me_category7_ann_B
    cat8 = me_category8_ann_A
    cat_eeA = empty_entry_A
    cat_eeB = empty_entry_B


    c_mat_1_1 = [[cat1], [cat1]]
    c_mat_1_nan = [[cat1], [cat_eeB]]
    c_mat_1_2 = [[cat1], [cat2]]

    c_mat_11_22 = [[cat1, cat2], [cat1, cat2]]
    c_mat_12_2nan = [[cat1, cat2], [cat2, cat_eeB]]
    c_mat_11_22_33 = [[cat1, cat2, cat3], [cat1, cat2,cat3]]
    c_mat_11_2nan_31 = [[cat1, cat2,cat3], [cat1, cat_eeB, cat1]]

    c_mat_5units_correct = [[cat1, cat2,cat3, cat1, cat2], [cat1, cat2,cat3, cat1, cat2]]
    c_mat_5units_malicious = [[cat1, cat_eeA,cat3, cat1, cat2], [cat1,cat3, cat_eeB, cat2, cat2]]

    c_mat_10units_correct = [[cat1, cat2,cat3, cat1, cat2, cat1, cat2,cat3, cat1, cat2], [cat1, cat2,cat3, cat1, cat2, cat1, cat2,cat3, cat1, cat2]]
    c_mat_10units_malicious = [[cat1, cat2, cat_eeA, cat1, cat2, cat1,cat6,cat8,cat3, cat_eeA], [cat_eeB, cat2, cat1, cat2, cat2,cat5,cat6,cat7,cat3, cat2]]

    c_mat_20units = [[cat1, cat2, cat_eeA, cat1, cat2, cat1,cat6,cat8,cat3, cat_eeA, cat1, cat2,cat3, cat1, cat2, cat1, cat2,cat3, cat1,cat5],
                     [cat_eeB, cat2, cat1, cat2, cat2,cat5,cat6,cat7,cat3, cat2, cat1, cat2,cat6, cat2,cat5, cat1, cat_eeB,cat3, cat1,cat5]]

    assert krippendorff_alpha.calculate_alpha(c_mat_1_1) == 1
    assert krippendorff_alpha.calculate_alpha(c_mat_1_nan) == 0
    assert krippendorff_alpha.calculate_alpha(c_mat_1_2) == 0
    assert krippendorff_alpha.calculate_alpha(c_mat_11_22) == 1
    assert krippendorff_alpha.calculate_alpha(c_mat_12_2nan) == -0.2
    assert krippendorff_alpha.calculate_alpha(c_mat_11_22_33) == 1
    assert krippendorff_alpha.calculate_alpha(c_mat_11_2nan_31) == 0.1667
    assert krippendorff_alpha.calculate_alpha(c_mat_5units_correct) == 1
    assert krippendorff_alpha.calculate_alpha(c_mat_5units_malicious) == 0.2703
    assert krippendorff_alpha.calculate_alpha(c_mat_10units_correct) == 1
    assert krippendorff_alpha.calculate_alpha(c_mat_10units_malicious) == 0.3049
    assert krippendorff_alpha.calculate_alpha(c_mat_20units) == 0.3953
