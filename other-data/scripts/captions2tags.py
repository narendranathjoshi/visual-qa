import json
from collections import defaultdict


def iterctr(items, n):
    ctr = 0
    for item in items:
        ctr += 1
        if ctr % n == 0:
            print(ctr)

        yield item

test_tags = list(filter(None, open("../test_tags.txt").read().splitlines()))

image_map = defaultdict(list)

data = json.load(open("../coco/captions_val2014.json"))

for ann in iterctr(data["annotations"], 10000):
    for tag in test_tags:
        if tag in ann["caption"]:
            image_map[tag].append(ann["image_id"])

image_map = sorted(image_map.items(), key=lambda x: len(x[1]), reverse=True)[:10]

# for k, v in image_map:
#     print(k, len(v))

json.dump(dict(image_map), open("sorted_val_img_map.json", "w"))
