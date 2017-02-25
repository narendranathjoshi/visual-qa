# Restricted datasets

This directory contains data for a subset of images from the visualQA dataset so that
we can run small experiments locally.

* `coco/` contains question and annotation data for 1000 images each of the training, testing,
and validation sets of VisualQA/MSCOCO.
The images are not included because they wouldn't fit in a GitHub repo. It also contains the
lists of selected images in each set.
* `scripts/` contains the scripts used to generate this subset.

## Scripts

### `extract_images.sh`

Download all 3 (training + testing + validation) VisualQA image zip files to one directory (for instance
`dir`. Yes I'm very creative.) Also copy the lists from the `coco/` directory into `dir`. Then
just run `sh extract_images.sh dir` and it'll extract the 3000 images from the zip archives.

### `select_json.py`

This generates the `.selected.json.gz` files from the question and annotation zip files provided
by VisualQA. Normally you don't need to use it since the results of running it are on the repo.

