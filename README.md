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

Run setup.py: `python setup.py develop`

## 3 Using the library

The main file `iaa.py` has 6 input arguments

`mode` = bbox or segm box depending on the type of annotation to evaluate

`result_destination` = path where the results are stored

`--annotation_format` = format of the annotations for which to evaluate the iaa - only coco is valid currently

`--iou_threshold` = values above this threshold are considered to be the same boxes/masks, default is 0.5

`--iaa_threshold` = values above this threshold are considered okay, all other are malicious, default is 0.6

`--filter` = filter by specific files or books, default/no filter is "" 

Example usage:

`python3 src/iaadet/calculate_iaa.py bbox src/landscape_annotations.json test_result`

## 4 Run tests

Navigate to parent folder and run:

`pytest -v`

## Cite us

```
@inproceedings{tschirschwitz2022texbig,
  title={A Dataset for Analysing Complex Document Layouts in the Digital Humanities and its Evaluation with Krippendorff ’s Alpha},
  author={Tschirschwitz, David and Klemstein, Franziska and Stein, Benno and Rodehorst, Volker},
  booktitle={DAGM German Conference on Pattern Recognition},
  year={2022},
  organization={Springer},
  note={(in press)}
}
```