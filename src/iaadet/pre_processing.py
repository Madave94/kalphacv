from collections import defaultdict

from iaadet import reliability_data

class Preprocess:
    def __init__(self, annotation_list):
        self.annotation_list = annotation_list
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
        all_images_lst = []
        for annotations in self.annotation_list:
            for current_image in annotations["images"]:
                all_images_lst.append( current_image["file_name"])
        all_images_set = set(all_images_lst)
        at_least_one_union_set = set()
        for image in all_images_set:
            lst = list(filter(lambda file_name: file_name == image, all_images_lst))
            if len(lst) > 1:
                at_least_one_union_set.add(image)
        return list(at_least_one_union_set)


    def sort_images(self):
        """
            Returns:
                dictonary containing:
                    key: file_name of a single image
                    value: list(annotations from all annotators coressponding to that key)
        """
        sorted_annotations = dict.fromkeys(self.image_set)
        for k, _ in sorted_annotations.items():
            sorted_annotations[k] = []

        for unique_image in self.image_set:
            for annotation in self.annotation_list:
                #index_unique_image = None
                for image in annotation["images"]:
                    if image['file_name'] in unique_image:
                        index_unique_image = image["id"]
                        for single_annotation in annotation["annotations"]:
                            if single_annotation["image_id"] == index_unique_image:
                                sorted_annotations[unique_image].append(single_annotation)
        return sorted_annotations

    def create_file_name_to_images_by_annotator_dict(self):
        """
            Create a dictionary with
                file_name as key
                list of all images from each annotator

            This relies on the filtered images that allow the calculation of an IAA, as returned by find_all_unique_images
        """
        image_name_to_images_by_annotator = defaultdict(list)
        for annotation in self.annotation_list:
            for image in annotation["images"]:
                image_name = image["file_name"]
                if image_name in self.image_set:
                    image_name_to_images_by_annotator[image_name].append(image)
        return image_name_to_images_by_annotator

    def find_all_categories(self):
        all_categories = {}

        for annotation in self.annotation_list:
            if 'categories' in annotation.keys():
                for category in annotation['categories']:
                    all_categories[category['id']] = category['name']
            return all_categories
        else:
            return {}
