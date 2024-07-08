import csv
import os

def to_csv_all(result_destination, image_list, iaa_threshold, iou_threshold, alpha_list):
    """
    Save all images with iaa value and a tag if image is malicious (iaa under threshold).
    @param result_destination: destination where csv file should be saved
    @param image_list: list of all images
    @param iaa_threshold: threshold of iaa values
    @param alpha_list: list of krippendorff alpha for every image

    """
    output_file = os.path.join(result_destination, "{}iou_all_results.csv".format(iou_threshold))
    header = ["file name", "iaa", "malicious"]
    with open(output_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for file_name, iaa in zip(image_list, alpha_list):
            data = [file_name, iaa]
            data.append("+") if iaa < iaa_threshold else data.append("")
            writer.writerow(data)
