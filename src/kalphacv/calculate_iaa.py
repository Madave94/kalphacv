import json
import os
from os import path
import argparse
from tqdm import tqdm

from kalphacv import pre_processing
from kalphacv import reliability_data
from kalphacv import krippendorff_alpha
from kalphacv import csv_data

def calculate_iaa(mode, unprocessed_annotations, images_rater_key, annotations_rater_key, iou_thresholds, iaa_threshold,
                  result_destination=None, silent=False):
    """
    main method for calculating the inter annotator agreement
    """

    preprocess_data = pre_processing.Preprocess(unprocessed_annotations, images_rater_key, annotations_rater_key, silent=silent)

    alpha_list = {iou_threshold: [] for iou_threshold in iou_thresholds}

    # for every image
    for image_name in tqdm(preprocess_data.image_names, desc=f"Calculating K-Alpha", disable=silent):
        # extract items from preprocessing
        annotations_by_image = preprocess_data.annotations_per_image[image_name]
        raters_by_image = preprocess_data.raters_per_image[image_name]
        size_for_image = preprocess_data.size_per_image[image_name]

        # construct reliability_data matrix
        rel_data = reliability_data.ReliabilityData(image_name, annotations_by_image, raters_by_image,
                                                    size_for_image)

        for iou_threshold in iou_thresholds:
            # matching of bboxes
            coincidence_matrix = rel_data.run(mode, iou_threshold)

            # bounding box iaa/krippendorff
            alpha = krippendorff_alpha.calculate_alpha(coincidence_matrix)
            alpha_list[iou_threshold].append(alpha)

    alpha_results = {}
    for iou_threshold in iou_thresholds:
        alpha = sum(alpha_list[iou_threshold]) / len(alpha_list[iou_threshold])
        alpha_results[iou_threshold] = alpha
        if result_destination is not None:
            csv_data.to_csv_all(result_destination, preprocess_data.image_names, iaa_threshold, iou_threshold, alpha_list[iou_threshold])
        elif not silent:
            print(f"{iou_threshold} IoU-Threshold: {alpha}")

    # should return alpha value here
    return alpha_results

def load_data(annotation_format, file_path, folder, filter, filter_empty=False):
    """
    setup all required data: parse input file to a dict and filter specific images if wanted
    """

    # Here other potential formats should be added and converted to the coco format
    if annotation_format == "coco":
        if folder:
            annotations = {
                "images": [],
                "annotations": []
            }
            # map id to new ids, to prevent overlapping ids between different json files
            image_id_counter = 1
            annotation_id_counter = 1
            for single_path in os.listdir(file_path):
                if "json" in single_path:
                    with open(file_path + "/" + single_path, "rb") as f:
                        data = json.load(f)
                    # map annotations to their label names to prevent different label counting from causing errors.
                    cat_id_to_label = {category["id"]: category["name"] for category in data["categories"]}
                    image_id_original_to_new = {}
                    for image in data["images"]:
                        image_id_original_to_new[image["id"]] = image_id_counter
                        image["id"] = image_id_counter
                        annotations["images"].append(image)
                        image_id_counter += 1
                    for annotation in data["annotations"]:
                        annotation["category_id"] = cat_id_to_label[ annotation["category_id"] ]
                        annotation["image_id"] = image_id_original_to_new[ annotation["image_id"] ]
                        annotation["id"] = annotation_id_counter
                        annotation_id_counter += 1
                        annotations["annotations"].append(annotation)
        else:
            with open(file_path, "rb") as f:
                annotations = json.load(f)

        # filter for specific images
        if filter != "":
            print(filter)
            new_images_lst = []
            image_ids = []
            new_annotations_lst = []
            for image in annotations["images"]:
                if filter in image["file_name"]:
                    new_images_lst.append(image)
                    image_ids.append(image["id"])
            for annotation in annotations["annotations"]:
                if annotation["image_id"] in image_ids:
                    new_annotations_lst.append(annotation)
            annotations["annotations"] = new_annotations_lst
            annotations["images"] = new_images_lst

        if filter_empty:
            image_ids_with_annotations = set()
            for annotation in annotations["annotations"]:
                image_ids_with_annotations.add( annotation["image_id"] )
            file_names_with_annotations = set()
            for image in annotations["images"]:
                if image["id"] in image_ids_with_annotations:
                    file_names_with_annotations.add( image["file_name"] )
            new_images_lst = []
            for image in annotations["images"]:
                if image["file_name"] in file_names_with_annotations:
                    new_images_lst.append( image )
            annotations["images"] = new_images_lst

        return annotations

def calculate_iaa_from_annotations(mode, source_annotation_path, images_rater_key=None, annotations_rater_key=None,
                                   result_destination=None, annotation_format="coco", folder=False,
                                   iou_thresholds: list = [0.5], iaa_threshold=0.6, filter="", filter_empty=False, silent=False):
    """
    Calculates the Inter-Annotator Agreement (IAA) based on the provided annotations and parameters.

    This function evaluates the agreement between different annotators using Krippendorff's Alpha,
    considering the provided annotations, IoU thresholds, and annotation format. It processes the
    annotations, matches bounding boxes or segmentation masks, and calculates agreement scores.

    Args:
        mode (str): The type of annotation to process, either "bbox" for bounding boxes or "segm"
            for segmentation masks.
        source_annotation_path (str): Path to the annotation file or directory containing multiple
            annotation files if `folder` is set to True.
        images_rater_key (str, optional): Key to identify annotators for images. If not provided,
            default keys (e.g., file_path or image_id) will be used.
        annotations_rater_key (str, optional): Key to identify annotators for individual annotations.
            If not provided, default keys will be used.
        result_destination (str, optional): Directory where the results in CSV format will be saved.
            If None, results will only be printed.
        annotation_format (str, optional): Format of the annotations to process. Default is "coco".
        folder (bool, optional): If True, processes all `.json` files in the provided directory instead
            of a single file. Default is False.
        iou_thresholds (list of float, optional): List of IoU thresholds to use for determining
            matches between annotations. Default is [0.5].
        iaa_threshold (float, optional): Threshold for acceptable agreement scores. Default is 0.6.
        filter (str, optional): A substring to filter specific files based on their names. Default is "".
        filter_empty (bool, optional): If True, filters out images with no annotations. Default is False.
        silent (bool, optional): If True, suppresses progress bars and print statements. Default is False.

    Returns:
        dict: A dictionary where keys are IoU thresholds and values are the corresponding
            Krippendorff's Alpha scores across the dataset.

    Raises:
        AssertionError: If the provided `source_annotation_path` does not exist.
        argparse.ArgumentTypeError: If IoU thresholds or the IAA threshold are not within the valid range.

    Notes:
        - Currently, only the "coco" annotation format is supported.
        - This function creates a reliability matrix for each image and calculates Krippendorff's Alpha
          based on the overlap of annotations using the specified IoU thresholds.
        - Results can be saved as CSV files if a destination directory is provided.
    """
    # load annotations
    annotations = load_data(
        annotation_format,
        source_annotation_path,
        folder,
        filter,
        filter_empty)

    # create results folder
    if result_destination is not None and not path.exists(result_destination):
        os.makedirs(result_destination)
        print("Created directory {}".format(result_destination))

    iaa = calculate_iaa(mode,
                  annotations,
                  images_rater_key,
                  annotations_rater_key,
                  result_destination=result_destination,
                  iou_thresholds=iou_thresholds,
                  iaa_threshold=iaa_threshold,
                  silent=silent)
    return iaa

def iou_threshold_type(value):
    value = float(value)
    if 0.0 < value <= 1.0:
        return value
    else:
        raise argparse.ArgumentTypeError(f"Invalid IoU threshold: {value}. Must be between 0 and 1.0 (exclusive of 0).")

def parse_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("mode", choices=["bbox", "segm"], help="select bounding box or polygons/instance segmentation", type=str)
    parser.add_argument("source_annotation_path", help="path to the annotation file or folder", type=str)

    # optional arguments used to find multi-annotations
    parser.add_argument("--images_rater_key", type=str, default=None, help="This key will be used to retrieve the information"
                                           "about who has annotated a single image/sample, if not set, the file_path and image_id will be used to determine it.")
    parser.add_argument("--annotations_rater_key", type=str, default=None, help="This key will be used to retrieve information"
                                           "about each annotation and who has annoated it, if not set, the file_path and image_id will be used to determine it.")

    # optional arguments
    valid_annotation_formats = ["coco"]
    parser.add_argument("--result_destination", help="place to store the results of the iaa in csv format", type=str)
    parser.add_argument("--annotation_format", choices=valid_annotation_formats, help="format of the annotations for which to evaluate the iaa", type=str, default="coco")
    parser.add_argument("--folder", help="Use all .json files in the folder instead of just a single annotation file - useful for extactions from CVAT for example", action="store_true")
    parser.set_defaults(folder=False)
    parser.add_argument("--iou_thresholds", help="values above this threshold are considered to be the same boxes/masks, if multiple are provided, the IAA will be calculated for each",
                        type=iou_threshold_type, nargs='+', default=[0.5])
    parser.add_argument("--iaa_threshold", help="values above this threshold are considered okay, all other are malicious", type=float, default=0.6)

    # filter - optional use
    parser.add_argument("--filter", help="add a filter to get only specific files", type=str, default="")
    parser.add_argument("--filter_empty", help="set this flag to filter all images that do not contain a single annotation", action="store_true")
    parser.set_defaults(filter_empty=False)

    # set silent flag for progress bar and print-out
    parser.add_argument("--silent", help="set this flag to omit print-outs.", action="store_true")
    parser.set_defaults(silent=False)

    args = parser.parse_args()

    # check files/folders exist
    assert path.exists(args.source_annotation_path), "Source annotation > {} < doesn't exist".format(args.source_annotation_path)

    # check thresholds in valid range
    # check thresholds in valid range
    for iou in args.iou_thresholds:
        assert 1.0 >= iou > 0.0, "IoU threshold needs to be between 1.0 (inclusive) or 0.0 (exclusive), current value is {}".format(
            iou)
    assert 1.0 >= args.iaa_threshold > -1.0, "IAA threshold needs to be between 1.0 (inclusive) or -1.0 (exclusive), " \
                                             "current value is {}".format(args.iaa_threshold)

    return args

if __name__ == '__main__':
    args = parse_arguments()
    calculate_iaa_from_annotations(args.mode, args.source_annotation_path, args.images_rater_key, args.annotations_rater_key,
                                   args.result_destination, args.annotation_format, args.folder,
                                   args.iou_thresholds, args.iaa_threshold, args.filter, args.filter_empty, args.silent)







