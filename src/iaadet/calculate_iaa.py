import json
import os
from os import path
import argparse

from iaadet import pre_processing
from iaadet import reliability_data
from iaadet import krippendorff_alpha
from iaadet import csv_data

def calculate_iaa(mode, annotation_list, result_destination, iou_threshold, iaa_threshold):
    """
    main method for calculating the inter annotator agreement
    """

    preprocess_data = pre_processing.Preprocess(annotation_list)

    alpha_list = []

    # for every image
    for image_name in preprocess_data.image_set:
        # extract items from preprocessing
        image_annotations = preprocess_data.sorted_annotations[image_name]
        image_name_to_images_by_annotator = preprocess_data.image_name_to_images_by_annotator[image_name]

        # construct reliability_data matrix
        rel_data = reliability_data.ReliabilityData(image_name, image_annotations,
                                                    image_name_to_images_by_annotator, iou_threshold)

        # matching of bboxes
        coincidence_matrix = rel_data.run(mode)

        # bounding box iaa/krippendorff
        alpha = krippendorff_alpha.calculate_alpha(coincidence_matrix)
        alpha_list.append(alpha)

    csv_data.to_csv_all(result_destination, preprocess_data.image_set, iaa_threshold, alpha_list)

def load_data(annotation_format, file_path, filter):
    """
    setup all required data: parse input file to a dict and filter specific images if wanted
    """

    # Here other potential formats should be added and converted to the coco format
    if annotation_format == "coco":
        annotations_lst = []
        with open(file_path, "rb") as f:
            annotations_lst.append(json.load(f))

        # filter for specific images
        if filter != "":
            print(filter)
            new_images_lst = []
            image_ids = []
            new_annotations_lst = []
            for image in annotations_lst[0]["images"]:
                if filter in image["file_name"]:
                    new_images_lst.append(image)
                    image_ids.append(image["id"])
            for annotation in annotations_lst[0]["annotations"]:
                if annotation["image_id"] in image_ids:
                    new_annotations_lst.append(annotation)
            annotations_lst[0]["annotations"] = new_annotations_lst
            annotations_lst[0]["images"] = new_images_lst

        return annotations_lst

def main(args):
    # load annotations
    source_annotation_path = args.source_annotation_path
    annotation_format = args.annotation_format
    annotation_list = load_data(
        annotation_format,
        source_annotation_path,
        args.filter)

    # creat results folder
    if not path.exists(args.result_destination):
        os.makedirs(args.result_destination)
        print("Created directory {}".format(args.result_destination))

    calculate_iaa(args.mode,
                  annotation_list,
                  result_destination= args.result_destination,
                  iou_threshold = args.iou_threshold,
                  iaa_threshold = args.iaa_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("mode", choices=["bbox", "segm"], help="select bounding box or polygons/instance segmentation", type=str)
    parser.add_argument("source_annotation_path", help="path to the annotations file", type=str)
    parser.add_argument("result_destination", help="place to store the results of the iaa in csv format", type=str)

    # optional arguments
    valid_annotation_formats = ["coco"]
    parser.add_argument("--annotation_format", choices=valid_annotation_formats, help="format of the annotations for which to evaluate the iaa", type=str, default="coco")
    parser.add_argument("--iou_threshold", help="values above this threshold are considered to be the same boxes/masks", type=float, default=0.5)
    parser.add_argument("--iaa_threshold", help="values above this threshold are considered okay, all other are malicious", type=float, default=0.6)

    # filter - optional use
    parser.add_argument("--filter", help="add a filter to get only specific files", type=str, default="")

    args = parser.parse_args()

    # check files/folders exist
    assert path.exists(args.source_annotation_path), "Source annotation > {} < doesn't exist".format(args.source_annotation_path)

    # check thresholds in valid range
    assert 1.0 >= args.iou_threshold > 0.0, "IoU threhols needs to be between 1.0 (inclusive) or 0.0 (exclusive), " \
                                            "current value is {}".format(args.iou_threshold)
    assert 1.0 >= args.iaa_threshold > -1.0, "IAA threhols needs to be between 1.0 (inclusive) or -1.0 (exclusive), " \
                                             "current value is {}".format(args.iaa_threshold)

    main(args)







