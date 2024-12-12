import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union

def calc_iou_bbox(bbox1, bbox2):
    """
    calculate the IoU for 2 bounding boxes
    """
    # method copied from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    if bbox1 == None or bbox2 == None:
        return 0
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


def calc_iou_segm_poly(segm1, segm2):
    """
    Calculate the IoU of two segmentation masks using Shapely.

    Parameters:
    - segm1, segm2: List of polygons, each represented as a list of coordinates.

    Returns:
    - IoU as a float.
    """
    if segm1 is None or segm2 is None:
        return 0.0

    # Function to convert list of coordinates into a list of tuples
    def coords_to_tuples(coords):
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # Convert the sub-shapes into Shapely polygons
    def create_polygons(coords_list):
        return [Polygon(coords_to_tuples(coords)) for coords in coords_list]

    def validate_polygon(geom):
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)  # Attempt to fix small invalidities
            if not geom.is_valid:
                geom = make_valid(geom)  # Attempt to fix further issues
            if not geom.is_valid:
                geom = geom.simplify(0.001, preserve_topology=True)  # Simplify if still invalid
            return geom
        except Exception as e:
            print(f"Executed non-topolgy preserving simplication {e}.")
            return geom.simplify(0.001, preserve_topology=False)  # Non-topology-preserving simplification as last resort

    # Create and validate polygons for both segmentations
    polygon1_shapes = [validate_polygon(p) for p in create_polygons(segm1)]
    polygon2_shapes = [validate_polygon(p) for p in create_polygons(segm2)]

    # Combine sub-shapes into a single (Multi)Polygon if necessary
    polygon1 = unary_union(polygon1_shapes) if len(polygon1_shapes) > 1 else polygon1_shapes[0]
    polygon2 = unary_union(polygon2_shapes) if len(polygon2_shapes) > 1 else polygon2_shapes[0]

    try:
        intersection = polygon1.intersection(polygon2)
        if intersection.is_empty or intersection.area == 0.0:
            return 0.0  # Early exit if there is no intersection
        union = polygon1.union(polygon2)
    except Exception as e:
        print(f"ShapelyError during intersection/union: {e}")
        return 0.0

    iou = intersection.area / union.area
    return iou

def calc_iou_segm_mask(entry1, entry2, image_size):
    mask1 = entry1.segm
    mask2 = entry2.segm
    bbox1 = entry1.bbox
    bbox2 = entry2.bbox

    if mask1 is None or mask2 is None:
        return 0.0

    bbox_iou = calc_iou_bbox(bbox1, bbox2)
    if bbox_iou == 0.0:
        return 0.0

    def adjust_bbox_and_region(mask, bbox, image_size):
        """Adjust bounding box to fit within image boundaries and prepare the mask."""
        x1_f, y1_f = bbox[0] * image_size[1], bbox[1] * image_size[0]
        x2_f, y2_f = (bbox[0] + bbox[2]) * image_size[1], (bbox[1] + bbox[3]) * image_size[0]

        x1, x2 = int(np.floor(x1_f)), int(np.ceil(x2_f))
        y1, y2 = int(np.floor(y1_f)), int(np.ceil(y2_f))

        # Adjust if there's a mismatch in broadcasting dimensions
        if x2 - x1 != mask.shape[1]:
            if abs(x1_f - x1) <= abs(x2 - x2_f):
                x1 = max(0, x1 - 1 if abs(x1_f - x1) > 1 else x1)
            else:
                x2 = min(image_size[1], x2 + 1 if abs(x2 - x2_f) > 1 else x2)

        if y2 - y1 != mask.shape[0]:
            if abs(y1_f - y1) <= abs(y2 - y2_f):
                y1 = max(0, y1 - 1 if abs(y1_f - y1) > 1 else y1)
            else:
                y2 = min(image_size[0], y2 + 1 if abs(y2 - y2_f) > 1 else y2)

        # Create the adjusted full-sized mask
        adjusted_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        adjusted_mask[:mask.shape[0], :mask.shape[1]] = mask

        return x1, y1, x2, y2, adjusted_mask

    # Adjust masks and bounding boxes
    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, adjusted_mask1 = adjust_bbox_and_region(mask1, bbox1, image_size)
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2, adjusted_mask2 = adjust_bbox_and_region(mask2, bbox2, image_size)

    # Place adjusted masks into full-sized image masks
    full_mask1 = np.zeros(image_size, dtype=bool)
    full_mask2 = np.zeros(image_size, dtype=bool)

    full_mask1[bbox1_y1:bbox1_y2, bbox1_x1:bbox1_x2] = adjusted_mask1
    full_mask2[bbox2_y1:bbox2_y2, bbox2_x1:bbox2_x2] = adjusted_mask2

    # Calculate intersection and union on full-sized masks
    intersection = np.logical_and(full_mask1, full_mask2).sum()
    union = np.logical_or(full_mask1, full_mask2).sum()

    if union == 0:  # Avoid division by zero
        return 0.0

    return intersection / union

def mask_to_array(seg, width, height):
    """
    helper function: convert segmentation mask to binary numpy array
    """
    arr_seg = Image.new('L', (width, height), 0)
    ImageDraw.Draw(arr_seg).polygon(seg, outline=1, fill=1)
    return np.array(arr_seg)