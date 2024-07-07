import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance

def calc_iou_bbox(bbox1, bbox2):
    """
    calculate the IoU for 2 bounding boxes
    """
    # method copied from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    boxA = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    boxB = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def calc_iou_seg(seg1, seg2):
    """
    calculate the IoU of two segmentation masks
    """
    # get width and height:
    seg1_max_x = max(seg1[::2])
    seg1_max_y = max(seg1[1::2])
    seg2_max_x = max(seg2[::2])
    seg2_max_y = max(seg2[1::2])

    width = max(seg1_max_x, seg2_max_x)+1
    height = max(seg1_max_y, seg2_max_y)+1

    mask1 = mask_to_array(seg1, width, height)
    mask2 = mask_to_array(seg2, width, height)

    # calculate IoU
    iou = 1-distance.jaccard(mask1.flatten(), mask2.flatten())

    return iou


def mask_to_array(seg, width, height):
    """
    helper function: convert segmentation mask to binary numpy array
    """
    arr_seg = Image.new('L', (width, height), 0)
    ImageDraw.Draw(arr_seg).polygon(seg, outline=1, fill=1)
    return np.array(arr_seg)