"""
    Algorithmic outline of the outer matching loop (example with three annotators):

    Legend:
        AE ~ empty entry of annotator A
        BE ~            "
        CE ~            "

    A = entries of annotator A = [A1, A2, A3]
    B = entries of annotator B = [B1, B2, B3]
    C = entries of annotator C = [C1, C2, C3]
    all_annotators = (A, B, C)

    Initialization process:
    matching_matrix = [ [A1, A2, A3]   # Set the elements of annotator A as matched elements
                        []
                        []          ]
    matched_annotators = (A) # take the first annotator of all_annotators
    unmatched_annotators =  (B, C) # the remaining annotators

    # Loop (the outer loop is a recursion in the implementation)
    for next_annotator_to_match in unmatched_annotators:
        unmatched_entries = get all unmatched entries of B which are not BE
        for matched_annotator in matched_annotators:
            available_entries = get all entries of A that are not AE and where B doesn't have a entry yet that is not BE

            matching_pairs = run matching between A and B returning pairs like (A1,B2) and (A3, B3) which are the matches

            update_matching_matrix = use the matching_pairs to fill the matrix, for example:
                                      [ [A1, A2, A3]
                                        [B2, BE, B3]
                                        []          ]

            unmatched_entries = now we use the pairs to update the newly still available entries [B1]

        # after running through all matched annotators the remaining unmatched entries will add a new column to everybody
        matching_matrix = [ [A1, A2, A3, AE]
                            [B2, BE, B3, B1]
                            []              ]
        matched_annotators = (A,B) update matched annotators with B
        unmatched_annotators = (C) update the unmatched annotators by removing the previously matched one
        # now the loop will be run again, but this time C can first match with A, all annotation that are not matched
        # with A can be still matched with B

        In short, C will try to match (C1,C2,C3) with (A1,A2,A3) but not AE, since AE is not a valid matching candidate
        Let's assume the matching pairs (A1,C2) and (A3,C3), which returns the following matching matrix:
        matching_matrix = [ [A1, A2, A3, AE]
                            [B2, BE, B3, B1]
                            [C2, CE, C3, CE]]
        Now trying to match C with B during available entries retrieval, only columns where B has a non empty entry and
        C has and empty entry are considered. Column 1 with B2 and C2 cannot be used since this elements have already
        matched with the assumption that if A1 matches B2 and A1 matches C2 then B2 also matches C2 (this might not be
        true, but we still assume this). The same condition holds for colum 3 with A3, B3 and C3. Column 2 does only have
        two emtpy entries, hence the condition that B needs to be non empty does not hold. Column 4 can attempt to match
        with all unmatched entries of C which are (C1), if C1 and B1 match the matching matrix looks as follows:
        matching_matrix = [ [A1, A2, A3, AE]
                            [B2, BE, B3, B1]
                            [C2, CE, C3, C1]]
"""


from collections import defaultdict
from iaadet import hungarian_matching

class MatrixEntry:
    def __init__(self, entry, annotator_name):
        self.bbox = entry["bbox"]
        self.category = entry["category_id"]
        self.annotator_id = entry["image_id"]
        self.id = entry["id"]
        self.attributes = entry["attributes"] if "attributes" in entry else None
        self.segmentation = entry["segmentation"] if "segmentation" in entry else None
        self.annotator_name = annotator_name

    def __repr__(self):
        return repr("ME " + str(self.id))

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

class EmptyEntry(MatrixEntry):
    def __init__(self, annotator_name):
        super(MatrixEntry, self).__init__()
        self.bbox = [0.0, 0.0, 0.0, 0.0]
        self.category = "*"
        self.id = -1
        self.annotator_id = 0
        self.attributes = {}
        self.annotator_name = annotator_name

    def __repr__(self):
        return "EE"

class ReliabilityData:
    def __init__(self, imageName, image_annotations, image_name_to_images_by_annotator, iou_threshold):
        self.imageName = imageName
        self.iou_threshold = iou_threshold
        self.id_to_annotator_names = { # this contains the image id to annotator name mapping
            img["id"]: img["annotator"] for img in image_name_to_images_by_annotator
        } # this short form is called a list/dictonary comprehension
        self.num_annotators = len(image_name_to_images_by_annotator)
        self.all_entries = self.add_entry_unites(image_annotations)

    def __call__(self, mode):
        """
            Call this function to create the coincidence matrix
        """
        # run matching algorithm with previously initialized parameter group
        # re initialize the dictionaries that will be worked on to prevent funky behaviour
        initial_annotator = self.get_lowest_element_in_set(self.id_to_annotator_names) # get first annotator
        matching_matrix = [[] for _ in range(self.num_annotators)] #
        matching_matrix[0] = list(self.all_entries[initial_annotator]) # list( entries_ann_A , [] , [] )
        return self.recursive_matching(matching_matrix, {initial_annotator}, mode ) # recursive call

    def run(self, mode):
        return self.__call__(mode)

    def get_lowest_element_in_set(self, annotator_set: set):
        """
            Convert a set to a list to sort it and get the lowest element in the set
        """
        lst = list(annotator_set)
        lst.sort()
        lowest_annotator = lst[0]
        return lowest_annotator

    def add_entry_unites(self, image_annotations):
        """
             creates a dictonary with the image id as key and a list of elements associated with this image id
             the image id correspondece to the annotator
        """
        all_entries = defaultdict(list)
        for annotation in image_annotations:
            annotator_name = self.id_to_annotator_names[annotation["image_id"]]
            entry = MatrixEntry(annotation, annotator_name)
            annotator_id = entry.annotator_id
            all_entries[annotator_id].append(entry)
        for entries in all_entries.values():
            entries.sort()
        return all_entries

    # recursive assignment
    def recursive_matching(self, matching_matrix, matched_annotators: set, mode):
        """
            See description on top of this file for the algorithm description
        """
        # set operation to find all unused annotator
        unmatched_annotators = set(self.id_to_annotator_names).difference(matched_annotators) # Set difference A - B
        if len(unmatched_annotators) == 0: # This criteria is needed to finish the recursion
            return matching_matrix
        else:
            current_row_len = len(matching_matrix[0])
            next_annotator_to_match = self.get_lowest_element_in_set(unmatched_annotators)  # get first annotator
            matching_matrix[len(matched_annotators)] = [EmptyEntry(self.id_to_annotator_names[next_annotator_to_match])] * current_row_len # add empty elements to fill length
            unmatched_entries = set(self.all_entries[next_annotator_to_match])
            for idx, matched_annotator_name in enumerate(matched_annotators):
                # run matching algorithm with all bboxes of matched_annotator_name and all bboxes of unmatched_bboxes of next_annotator_to_match
                available_entries_old_ann = self.positional_retrieval(matching_matrix, idx, len(matched_annotators))

                # run matching
                if len(available_entries_old_ann) > 0 and len(unmatched_entries) > 0:
                    matched_pairs = hungarian_matching.run_matching(available_entries_old_ann, unmatched_entries, self.iou_threshold, mode)
                else:
                    matched_pairs = []

                # upgrade matrix with the matching pairs
                matching_matrix = self.positional_upgrade(matching_matrix, matched_pairs, idx, len(matched_annotators))

                # update remaining unmatched boxes of new annotator
                if matched_pairs is not None:
                    newly_matched = [entry for _, entry in matched_pairs]
                    unmatched_entries = unmatched_entries.difference(newly_matched)

                # if all boxes could be matched allow earlier to return current matching table
                # it also doesn't require to
                if len(unmatched_entries) == 0:
                    matched_annotators.add(next_annotator_to_match)
                    return self.recursive_matching(matching_matrix, matched_annotators, mode)
            matching_matrix = self.add_unmatched_entries_to_new_rows(matching_matrix, matched_annotators, unmatched_entries)
            matched_annotators.add(next_annotator_to_match)
            return self.recursive_matching(matching_matrix, matched_annotators, mode)

    def positional_retrieval(self, matching_matrix, matched_annotator_idx, unmatched_annotator_idx):
        """
            Function to get the available entries for matching with the new annotator
        """
        free_entries = set()
        for col, entry in enumerate(matching_matrix[unmatched_annotator_idx]):
            if isinstance(entry, EmptyEntry):
                entry_to_consider = matching_matrix[matched_annotator_idx][col]
                if not isinstance(entry_to_consider, EmptyEntry):
                    free_entries.add(
                        matching_matrix[matched_annotator_idx][col]
                    )
        return free_entries

    def positional_upgrade(self, matching_matrix, matched_pairs, matched_annotator_idx, unmatched_annotator_idx):
        """
            Function to add the matched pairs to the matching matrix using the index positions of the two annotators
        """
        if matched_pairs is not None:
            for entry_old, entry_new in matched_pairs:
                col = matching_matrix[matched_annotator_idx].index(entry_old)
                matching_matrix[unmatched_annotator_idx][col] = entry_new
        return matching_matrix

    def add_unmatched_entries_to_new_rows(self, matching_matrix, matched_annotators, unmatched_entries: set):
        """
            After attempting to match all entries with the iterative hungarian matching, this function create one new
            column for each non-matched entry, it will also creat empty entries for all the previous annotators.
        """
        matched_annotators = list(matched_annotators)
        unmatched_annotator_idx = len(matched_annotators)
        unmatched_entries = list(unmatched_entries)
        unmatched_entries.sort()
        number_of_cols_to_add = len(unmatched_entries)
        for matrix_row in range(unmatched_annotator_idx+1):
            if matrix_row == unmatched_annotator_idx:
                matching_matrix[matrix_row] += unmatched_entries
            else:
                if len(matching_matrix[matrix_row]) == 0:
                    annotator_name = matched_annotators[matrix_row]
                else:
                    annotator_name = matching_matrix[matrix_row][0].annotator_name # get annotator name
                matching_matrix[matrix_row] += [EmptyEntry(annotator_name)] * number_of_cols_to_add
        return matching_matrix
