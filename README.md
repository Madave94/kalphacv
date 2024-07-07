# Inter Annotator Agreement

This is the code for evaluating Krippendorffs Alpha for object detection or instance segemantation as explained in our paper (see at the bottom)
.
## 1 Installation

### Clone via ssh
 `git@github.com:Madave94/KrippendorffAlphaComputerVision.git`
### Clone via https
`https://github.com/Madave94/KrippendorffAlphaComputerVision.git`
## 2 Create and activate virtual environment

Enter folder: `cd inter_annotator_agreement`

Create virtual environment: `python3 -m venv iaa_env`

Activate virtual environment: `source iaa_env/bin/activate`

Run setup.py: `pip install .`

## 3 Using the library

### Formatting your COCO-JSON

There are two requirements that your annotation file needs to fulfill so that it can calculate the IAA from the COCO-JSON.

1. Files/Images annotated by different annotators need to have different image IDs. This is logical since otherwise it would not be possible to map the annotations to annotators.
2. You need to provide an "annotator" key and a value for the annotator that annotated the image for each annotator.

A single entry in the images list might look like this:
```
{
    'width': 640,                     # Image width
    'height': 427,                    # Image height
    'file_name': '000000397133.jpg',  # File name of the image
    'id': 397133,                     # Unique image ID
    'annotator': 'coder_a'            # Annotator identifier
}
```

It can contain additional information. In case you do not have any coder information, just split into generic groups.


### Command Line Usage

The main file `calculate_iaa.py` has 6 input arguments, with `mode` and `source_annotation_path` being required:

1. **`mode`** (required)  
   Type of annotation to evaluate: `bbox` or `segm`.

2. **`source_annotation_path`** (required)  
   Path to the annotation file.

3. **`--result_destination`** (optional)  
   Path where the results are stored.

4. **`--annotation_format`** (optional)  
   Format of the annotations for which to evaluate the IAA. Only `coco` is valid currently. Default is `coco`.

5. **`--iou_threshold`** (optional)  
   Values above this threshold are considered to be the same boxes/masks. Default is `0.5`.

6. **`--iaa_threshold`** (optional)  
   Values above this threshold are considered okay; all others are malicious. Default is `0.6`.

7. **`--filter`** (optional)  
   Filter by specific files or books. Default is `""` (no filter).

8. **`--filter_empty`** (optional)  
   Set this flag to filter all images that do not contain a single annotation.

### Example Usage:

To evaluate bounding box annotations using the `calculate_iaa.py` script, you can run the following command:

```sh
python3 src/iaadet/calculate_iaa.py bbox src/landscape_annotations.json
```

### Python API Usage

If you are importing the package and using the `calculate_iaa_from_annotations` function directly in your code, you can do so as follows:

```
from kalphacv import calculate_iaa_from_annotations

calculate_iaa_from_annotations(
    mode="bbox",
    source_annotation_path="path/to/annotations.json",
    result_destination="path/to/results",
    annotation_format="coco",
    iou_threshold=0.5,
    iaa_threshold=0.6,
    filter="",
    filter_empty=False
)
```


## Cite us

```
@inproceedings{10.1007/978-3-031-16788-1_22,
  title={A Dataset for Analysing Complex Document Layouts in the Digital Humanities and its Evaluation with Krippendorff ’s Alpha},
  author={Tschirschwitz, David and Klemstein, Franziska and Stein, Benno and Rodehorst, Volker},
  booktitle="Pattern Recognition",
  year={2022},
  publisher="Springer International Publishing",
  address="Cham",
  pages="354--374",
}
```
