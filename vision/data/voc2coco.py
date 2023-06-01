import argparse
import json
import os
from glob import glob
import xml.etree.ElementTree as elemTree

from tqdm import tqdm


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext("name")

    if label not in label2id:
        label2id[label] = len(label2id) + 1

    category_id = label2id[label]
    bndbox = obj.find("bndbox")
    xmin = float(bndbox.findtext("xmin")) - 1
    ymin = float(bndbox.findtext("ymin")) - 1
    xmax = float(bndbox.findtext("xmax"))
    ymax = float(bndbox.findtext("ymax"))
    assert (
        xmax > xmin and ymax > ymin
    ), f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        "area": o_width * o_height,
        "iscrowd": 0,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "ignore": 0,
        "segmentation": [],  # This script is not for segmentation
    }
    return ann


def get_image_info(annotation_root):
    path = annotation_root.findtext("path")
    if path is None:
        filename = annotation_root.findtext("filename")
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]

    size = annotation_root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    image_info = {"file_name": filename, "height": height, "width": width, "id": img_id}
    return image_info


if __name__ == "__main__":
    """
    example
        python ./vision/data/voc2coco.py --annotation_dir ../datasets/voc/VOC2012_train_val/Annotations \
                --output_label ../datasets/voc/VOC2012_train_val/coco_label.json
        python ./vision/data/voc2coco.py --annotation_dir ../datasets/voc/VOC2012_train_val/Annotations \
                --output_label ../datasets/voc/VOC2012_train_val/coco_label.json
    """
    config = argparse.ArgumentParser()

    config.add_argument("--annotation_dir", type=str)
    config.add_argument("--output_label", type=str)

    args = config.parse_args()

    paths = glob(os.path.join(args.annotation_dir, "*"))

    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }
    bnd_id = 1
    label2id = {}

    for path in tqdm(paths, total=len(paths)):
        annotation_root = elemTree.parse(path)

        img_info = get_image_info(annotation_root=annotation_root)
        img_id = img_info["id"]
        output_json_dict["images"].append(img_info)

        for obj in annotation_root.findall("object"):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({"image_id": img_id, "id": bnd_id})
            output_json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for k, v in label2id.items():
        output_json_dict["categories"] += [{"id": v, "name": k}]

    with open(args.output_label, "w") as f:
        json.dump(output_json_dict, f)
