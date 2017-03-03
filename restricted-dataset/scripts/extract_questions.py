#!/usr/bin/env python3

import json
import re
import sys
import gzip

# Pass the script the names of all the files it should extract questions from.

if len(sys.argv) < 2:
    print("  ! Error: no input file specified")
    sys.exit(1)

for input_file_name in sys.argv[1:]:
    with gzip.open(input_file_name) as input_file:
        data = json.load(input_file)
        if 'questions' not in data.keys():
            print("\n  ! Error: unrecognized file type")

        output_file_elts = input_file_name.split(".")
        output_file_elts[-1] = "question-text.gz"
        output_file_name = ".".join(output_file_elts)
        print("Writing to {}".format(output_file_name))
        with gzip.open(output_file_name, 'wt') as output_file:
            for q in data["questions"]:
                print(q["question"], file=output_file)

