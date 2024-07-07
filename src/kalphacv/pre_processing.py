from collections import defaultdict

class Preprocess:
    def __init__(self, annotations):
        self.annotations = annotations
        self.image_set = self.find_all_unique_images()
        self.sorted_annotations = self.sort_images()
        self.image_name_to_images_by_annotator = self.create_file_name_to_images_by_annotator_dict()
        self.category_list = self.find_all_categories()

    def find_all_unique_images(self):
        """
            Returns a set of all uniquly existing image that have been at least annotated by two annotators
            At least annotated by two annotators can also mean that one annotator did not make a annotation on the page,
            so intended to leave the page blank.
            -> meaning all images where an IAA can be calculated
        """
        file_name_count = defaultdict(int)
        for current_image in self.annotations["images"]:
            file_name_count[current_image["file_name"]] += 1

        # Extract file names that appear more than once
        duplicates = [file_name for file_name, count in file_name_count.items() if count > 1]
        return duplicates


    def sort_images(self):
        """
            Returns:
                dictonary containing:
                    key: file_name of a single image
                    value: list(annotations from all annotators coressponding to that key)
        """
        sorted_annotations = {key: [] for key in self.image_set}

        included_image_ids = []
        image_id_to_file_name = {}
        for image in self.annotations["images"]:
            if image["file_name"] in self.image_set:
                included_image_ids.append(image["id"])
                image_id_to_file_name[image["id"]] = image["file_name"]

        for single_annotation in self.annotations["annotations"]:
            # check if this image is part of the collection of image_ids
            if single_annotation["image_id"] in included_image_ids:
                file_name = image_id_to_file_name[single_annotation["image_id"]]
                sorted_annotations[file_name].append(single_annotation)

        return sorted_annotations

    def create_file_name_to_images_by_annotator_dict(self):
        """
            Create a dictionary with
                file_name as key
                list of all images from each annotator

            This relies on the filtered images that allow the calculation of an IAA, as returned by find_all_unique_images
        """
        image_name_to_images_by_annotator = defaultdict(list)
        for image in self.annotations["images"]:
            image_name = image["file_name"]
            if image_name in self.image_set:
                image_name_to_images_by_annotator[image_name].append(image)
        return image_name_to_images_by_annotator

    def find_all_categories(self):
        all_categories = {}
        if 'categories' in self.annotations.keys():
            for category in self.annotations['categories']:
                all_categories[category['id']] = category['name']
            return all_categories
        else:
            return {}
