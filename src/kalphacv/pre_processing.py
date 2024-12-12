from collections import defaultdict
from tqdm import tqdm

class Preprocess:
    """
       Processes the dataset to get the canonical annotation format including rater_list and rater as attributes for
       images and annotations.
    """
    def __init__(self, unprocessed_annotations, images_rater_key, annotations_rater_key, silent):
        self.silent = silent
        self.image_names = None
        self.raters_per_image = None
        self.annotations_per_image = None
        self.size_per_image = None
        self.preprocess_annotations(unprocessed_annotations, images_rater_key, annotations_rater_key)

    def preprocess_annotations(self, unprocessed_annotations, images_rater_key, annotations_rater_key):
        image_names = []
        raters_per_image = {}
        annotations_per_image = defaultdict(list)
        image_id_to_image_name = {}
        image_size = {}
        ### Case 1 - both keys are there
        if images_rater_key and annotations_rater_key:
            for image in unprocessed_annotations["images"]:
                image_name = image["file_name"]
                image_names.append(image_name)
                image_id_to_image_name[image["id"]] = image_name
                raters = image[images_rater_key]
                raters_per_image[image["file_name"]] = raters
                image["rater_list"] = raters # add field, might be used later
                image_size[image_name] = (image["height"], image["width"])
            for annotation in unprocessed_annotations["annotations"]:
                image_name = image_id_to_image_name[annotation["image_id"]]
                rater = annotation[annotations_rater_key]
                annotation["rater"] = rater
                annotations_per_image[image_name].append(annotation)
        ### Case 2 - neither key is set
        elif not images_rater_key and not annotations_rater_key:
            image_id_to_file_path = {}
            for image in unprocessed_annotations["images"]:
                file_path = image["file_name"]
                image_id = image["id"]
                image_id_to_file_path[image_id] = (file_path)
                if file_path in image_names:
                    raters_per_image[file_path].append(image_id)
                else:
                    image_names.append(file_path)
                    raters_per_image[file_path] = [image_id]
                    image_size[file_path] = (image["height"], image["width"])
            for annotation in unprocessed_annotations["annotations"]:
                image_id = annotation["image_id"]
                annotation["rater"] = image_id
                annotations_per_image[ image_id_to_file_path[image_id] ].append(annotation)
        else:
            raise Exception("This case is not supported")

        self.image_names = image_names
        self.raters_per_image = raters_per_image
        self.annotations_per_image = annotations_per_image
        self.size_per_image = image_size



