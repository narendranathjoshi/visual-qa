#!/usr/bin/env python3

import zipfile
import json
import re
import sys
import gzip


## Helper class to read a json file from a zip file
class DecodeReader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, num=None):
        return self.stream.read(num).decode()


## Helper function to filter relevant questions/annotations
def relevant(q, selected_ids):
    return q['image_id'] in selected_ids


# 1 - Build the list of image IDs
selection_files = ['test_images_select.txt', 'train_images_select.txt', 'val_images_select.txt']
selected_ids = []

id_re = re.compile("_([0-9]+)\.jpg")

for fname in selection_files:
    with open(fname) as sel_file:
        for line in sel_file:
            selected_ids.append(int(id_re.search(line).groups()[0]))

print("  * Selected IDs from {}".format(", ".join(selection_files)))

# 2 - Open the specified file, keep only the useful part, then write it as a
#     gzipped file.

if len(sys.argv) != 2:
    print("  ! Error: No input file specified")
    sys.exit(1)

zfile = zipfile.ZipFile(sys.argv[1])
names = zfile.namelist()

print("  * Found {} files".format(len(names)))

for name in names:
    print("  > {} ".format(name), end="")
    jsonfile = DecodeReader(zfile.open(name))
    data = json.load(jsonfile)
    if 'annotations' in data.keys():
        field = "annotations"
    elif 'questions' in data.keys():
        field = "questions"
    else:
        print("\n  ! Error: unrecognized file type")
        continue

    print(" ({})".format(field))

    data[field] = [x for x in data[field] if relevant(x, selected_ids)]
    print("    Selected {} relevant {}".format(len(data[field]), field))

    out_name = ".".join(name.split(".")[:-1]) + ".selected.json.gz"
    print("  < {}".format(out_name))
    
    with gzip.open(out_name, 'wt') as out_file:
        json.dump(data, out_file, indent=4)

