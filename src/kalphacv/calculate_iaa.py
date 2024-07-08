import json
import os
from os import path
import argparse
from tqdm import tqdm

from kalphacv import pre_processing
from kalphacv import reliability_data
from kalphacv import krippendorff_alpha
from kalphacv import csv_data

def calculate_iaa(mode, annotations, iou_thresholds, iaa_threshold, result_destination=None, silent=False):
    """
    main method for calculating the inter annotator agreement
    """

    preprocess_data = pre_processing.Preprocess(annotations, silent=silent)

    alpha_list = {iou_threshold: [] for iou_threshold in iou_thresholds}

    # for every image
    for image_name in tqdm(preprocess_data.image_set, desc=f"4/4 - Calculating K-Alpha", disable=silent):
        # extract items from preprocessing
        image_annotations = preprocess_data.sorted_annotations[image_name]
        image_name_to_images_by_annotator = preprocess_data.image_name_to_images_by_annotator[image_name]

        for iou_threshold in iou_thresholds:
            # construct reliability_data matrix
            rel_data = reliability_data.ReliabilityData(image_name, image_annotations,
                                                        image_name_to_images_by_annotator, iou_threshold)

            # matching of bboxes
            coincidence_matrix = rel_data.run(mode)

            # bounding box iaa/krippendorff
            alpha = krippendorff_alpha.calculate_alpha(coincidence_matrix)
            alpha_list[iou_threshold].append(alpha)

    alpha_results = {}
    for iou_threshold in iou_thresholds:
        alpha = sum(alpha_list[iou_threshold]) / len(alpha_list[iou_threshold])
        alpha_results[iou_threshold] = alpha
        if result_destination is not None:
            csv_data.to_csv_all(result_destination, preprocess_data.image_set, iaa_threshold, iou_threshold, alpha_list)
        elif not silent:
            print(f"{iou_threshold} IoU-Threshold: {alpha}")

    # should return alpha value here
    return alpha_results

def load_data(annotation_format, file_path, filter, filter_empty=False):
    """
    setup all required data: parse input file to a dict and filter specific images if wanted
    """

    # Here other potential formats should be added and converted to the coco format
    if annotation_format == "coco":
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

def calculate_iaa_from_annotations(mode, source_annotation_path, result_destination=None, annotation_format="coco",
                                   iou_thresholds: list = [0.5], iaa_threshold=0.6, filter="", filter_empty=False, silent=False):
    # load annotations
    annotations = load_data(
        annotation_format,
        source_annotation_path,
        filter,
        filter_empty)

    # create results folder
    if result_destination is not None and not path.exists(result_destination):
        os.makedirs(result_destination)
        print("Created directory {}".format(result_destination))

    iaa = calculate_iaa(mode,
                  annotations,
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
    parser.add_argument("source_annotation_path", help="path to the annotation file", type=str)

    # optional arguments
    valid_annotation_formats = ["coco"]
    parser.add_argument("--result_destination", help="place to store the results of the iaa in csv format", type=str)
    parser.add_argument("--annotation_format", choices=valid_annotation_formats, help="format of the annotations for which to evaluate the iaa", type=str, default="coco")
    parser.add_argument("--iou_thresholds", help="values above this threshold are considered to be the same boxes/masks",
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
    calculate_iaa_from_annotations(args.mode, args.source_annotation_path, args.result_destination, args.annotation_format,
                                   args.iou_thresholds, args.iaa_threshold, args.filter, args.filter_empty, args.silent)







