# Inter Annotator Agreement (IAA) in Computer Vision

This is the code for evaluating Krippendorff's Alpha for object detection and instance segmentation as explained in our 
paper (see at the bottom).

## 1 Installation

### Clone via ssh
 `git clone git@github.com:Madave94/kalphacv.git`
### Clone via https
`git clone https://github.com/Madave94/kalphacv.git`
## 2 Create and activate virtual environment

Enter folder: `cd kalphacv`

Create virtual environment: `python3 -m venv iaa_env`

Activate virtual environment: `source iaa_env/bin/activate`

Run setup.py: `pip install .`

## 3 Using the library

The library is currently focused on calculating the inter annotator agreement (IAA) for:

- Bounding Boxes (Object Detection in xywh format)
- Segmentation Polygons (Instance Segmentation as used in COCO-Format)
- [Not Recommended] Segmentation Masks (Instance Segmentation using rastarization of the labels)

### 3.1 Target annotation format

The data will be move to a canonical format that only contains the minimum information necessary to compute K-Alpha (IAA).
To see which kind of annotations are accepted see chapter 3.2. For each image these are the following information:

```
   {
        "file_name": '000000397133.jpg',    # File name of the image
        "width": 640,                       # Image width
        "height": 427,                      # Image height
        "id": 397133,                       # Unique image ID
        "rater_list": ["r1", "r2"]          # List of annnotators assigned to label this image
   }
```

For each annotation these are the following information

```
    {
        "image_id": 397133,                 # Image ID corresponding to the ID mentioned in an image inside the dataset
        "category_id": 199,                 # Class ID or class name, both is allowed
        "bbox": [100, 50, 45, 30]           # Bounding Box in the form XYWH as for COCO
        "segmentation": [[100, 50, ...]]    # Polygon in format X1Y1, X2Y2. Allows multiple shapes for complex shapes
        "rater": "r2"                       # Rater identification that annotated this instance
    }
```

### 3.2 Accepted annotation formats

#### Option a (Recommended)

Provide the annotations as described above with the two key-value pairs `images` and `annotations` each containing a list
of dictionaries.

For multi-annotated data, two pieces of information are important:
- Who was assigned to annotate an image? -> represented as the rater_list
- Which instance was create by which annotator? -> shown by the rater

#### Option b 

**Background**: In case this information is not available it will be possible to still calculate the inter-annotator agreement using
generic groups. This however, will not allow evaluations of annotator-vitality or other properties. For such a case it
is sufficient to provide one or multiple files. 

Provide COCO-Formated annotations either in a single file or in a single folder. The files will be automatically split up. 
The different `image_id`'s will be used to look up which annotation belongs to whom.

*Common use case*: If you extract annotations from CVAT you can use this directly.

### 3.3 CLI Usage (call from the command line)

Get help for possible command line options:
```
python src/kalphacv/calculate_iaa.py --help
```
Example call:
```
python bbox /path/to/annotation/folder/ --folder
```

### 3.4 API Usage (call from within another python script)

If you are importing the package and using the `calculate_iaa_from_annotations` function directly in your code, you can do so as follows:

```
from kalphacv import calculate_iaa_from_annotations

# Example usage
iaa_results = calculate_iaa_from_annotations(
    mode="bbox",                                        # Specify "bbox" for bounding boxes or "segm" for segmentation masks
    source_annotation_path="path/to/annotations.json",  # Path to the annotation file or directory
    images_rater_key=None,                              # Optional: Key to identify annotators for images
    annotations_rater_key=None,                         # Optional: Key to identify annotators for individual annotations
    result_destination="path/to/results",               # Directory to save results in CSV format (optional)
    annotation_format="coco",                           # Specify the annotation format, e.g., "coco"
    folder=False,                                       # Set to True if processing multiple files in a folder
    iou_thresholds=[0.5],                               # List of IoU thresholds to evaluate agreement
    iaa_threshold=0.6,                                  # Threshold for acceptable agreement
    filter="",                                          # Optional: Substring to filter files based on their names
    filter_empty=False,                                 # Set to True to exclude images with no annotations
    silent=False                                        # Set to True to suppress print statements and progress bars
)

# The `iaa_results` dictionary will contain IoU thresholds as keys and their respective
# Krippendorff's Alpha scores as values:
# Example: {0.5: 0.85, 0.75: 0.78}
print(iaa_results)

```


## Cite us

```
@inproceedings{tschirschwitz2022,
  title={A Dataset for Analysing Complex Document Layouts in the Digital Humanities and its Evaluation with Krippendorff ’s Alpha},
  author={Tschirschwitz, David and Klemstein, Franziska and Stein, Benno and Rodehorst, Volker},
  booktitle={Proceedings of the German Conference on Pattern Recognition (GCPR)},
  year={2022},
  doi = {10.1007/978-3-031-16788-1_22}
}
```
