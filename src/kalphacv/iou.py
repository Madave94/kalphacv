import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.errors import TopologicalError
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


def calc_iou_seg(segm1, segm2):
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


def mask_to_array(seg, width, height):
    """
    helper function: convert segmentation mask to binary numpy array
    """
    arr_seg = Image.new('L', (width, height), 0)
    ImageDraw.Draw(arr_seg).polygon(seg, outline=1, fill=1)
    return np.array(arr_seg)