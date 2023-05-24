import copy
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor


class COCOEval:
    def __init__(self, label_path: str):
        self.coco_gt = COCO(label_path)
        self.coco_dt = COCO()

        self.coco_dt.dataset["categories"] = copy.deepcopy(
            self.coco_gt.dataset["categories"]
        )
        self.coco_dt.dataset["annotations"] = []
        self.anno_id = 1

    def update_dt(
        self, image_ids: Tensor, category_ids: Tensor, boxes: Tensor, scores: Tensor
    ):
        image_ids = image_ids.cpu().tolist()
        category_ids = category_ids.cpu().tolist()
        boxes = boxes.cpu().tolist()
        scores = scores.cpu().tolist()

        for image_id, category_id, box, score in zip(
            image_ids, category_ids, boxes, scores
        ):
            x1, x2, y1, y2 = [box[0], box[0] + box[2], box[1], box[1] + box[3]]
            self.coco_dt.dataset["annotations"] += [
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "box": box,
                    "score": score,
                    "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                    "area": box[2] * box[3],
                    "id": self.anno_id,
                    "iscrowd": 0,
                }
            ]
            self.anno_id += 1

    def eval(self) -> Dict[str, float]:
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_stats = coco_eval.stats

        result = {}

        if coco_stats:
            result["coco/mAP_50:95"] = coco_stats[0]
            result["coco/mAP_50"] = coco_stats[1]
            result["coco/mAR_50:95"] = coco_stats[8]

        return result
