import json
from collections import defaultdict


def iterctr(items, n):
    ctr = 0
    for item in items:
        ctr += 1
        if ctr % n == 0:
            print(ctr)

        yield item

test_tags = ["cat", "dog", "water", "room", "sign", "shirt", "bus", "plate",
             "can", "food", "train", "girl", "table", "red", "plane", "boy"]

image_map = defaultdict(list)

data = json.load(open("../coco/captions_val2014.json"))

for ann in iterctr(data["annotations"], 10000):
    for tag in test_tags:
        if tag in ann["caption"]:
            image_map[tag].append(ann["image_id"])


json.dump(image_map, open("val_img_map.json", "w"))
