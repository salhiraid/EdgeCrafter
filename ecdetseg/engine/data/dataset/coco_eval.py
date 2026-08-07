"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os
import pickle

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ...core import register
from ...misc import dist_utils

__all__ = ['CocoEvaluator',]

COCO_METRIC_NAMES = {
    'bbox': (
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large',
    ),
    'segm': (
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large',
    ),
    'keypoints': (
        'AP', 'AP50', 'AP75', 'AP_medium', 'AP_large',
        'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large',
    ),
    # Keep the default pixel-distance names in the shared mapping as well as
    # generating them dynamically in ``metric_names``.  This makes the
    # TensorBoard/evaluation path compatible with older evaluator instances
    # whose ``metric_names`` method only indexes this mapping.
    'pose': (
        'Precision_5px', 'Recall_5px', 'F1_5px',
        'Precision_10px', 'Recall_10px', 'F1_10px',
        'Visibility_Precision', 'Visibility_Recall', 'Visibility_F1',
    ),
}


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, verbose=True, keypoint_oks_sigmas=None,
                 keypoint_score_mode='bbox_keypoint', keypoint_score_thr=0.2,
                 keypoint_distance_thresholds=(5.0, 10.0),
                 keypoint_visibility_thr=0.5, keypoint_match_iou_thr=0.5,
                 pose_detection_score_thr=0.3):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.coco_gt.dataset.setdefault('info', {})
        self.iou_types = iou_types
        self.keypoint_oks_sigmas = keypoint_oks_sigmas
        allowed_score_modes = ('bbox', 'keypoint', 'bbox_keypoint')
        if keypoint_score_mode not in allowed_score_modes:
            raise ValueError(
                f'keypoint_score_mode must be one of {allowed_score_modes}, got {keypoint_score_mode!r}')
        self.keypoint_score_mode = keypoint_score_mode
        self.keypoint_score_thr = float(keypoint_score_thr)
        self.keypoint_distance_thresholds = tuple(float(v) for v in keypoint_distance_thresholds)
        if not self.keypoint_distance_thresholds or any(v <= 0 for v in self.keypoint_distance_thresholds):
            raise ValueError('keypoint_distance_thresholds must contain positive pixel distances')
        self.keypoint_visibility_thr = float(keypoint_visibility_thr)
        self.keypoint_match_iou_thr = float(keypoint_match_iou_thr)
        self.pose_detection_score_thr = float(pose_detection_score_thr)
        if "keypoints" in iou_types:
            self._prepare_keypoint_ground_truth()
        self.labels = [cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())] if verbose else None

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = self._make_coco_eval(iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self._reset_pose_counts()

    def _prepare_keypoint_ground_truth(self):
        """Make bbox-only annotations valid ignored entries for COCO keypoint eval."""
        fallback_keypoints = len(self.keypoint_oks_sigmas or [])
        category_keypoints = {
            category['id']: len(category.get('keypoints', [])) or fallback_keypoints
            for category in self.coco_gt.dataset.get('categories', [])
        }
        for annotation in self.coco_gt.dataset.get('annotations', []):
            num_keypoints = category_keypoints.get(annotation.get('category_id'), fallback_keypoints)
            annotation['_keypoint_valid'] = bool(annotation.get('keypoints'))
            if not annotation.get('keypoints'):
                annotation['keypoints'] = [0.0] * (num_keypoints * 3)
                annotation['num_keypoints'] = 0
            else:
                annotation['num_keypoints'] = int(annotation.get(
                    'num_keypoints', sum(v > 0 for v in annotation['keypoints'][2::3])))
        self.coco_gt.createIndex()

    def _make_coco_eval(self, iou_type):
        coco_eval = COCOeval(self.coco_gt, iouType=iou_type)
        if iou_type == 'keypoints' and self.keypoint_oks_sigmas is not None:
            sigmas = np.asarray(self.keypoint_oks_sigmas, dtype=np.float64)
            expected = max(
                (len(category.get('keypoints', [])) for category in self.coco_gt.dataset.get('categories', [])),
                default=0,
            )
            if expected and len(sigmas) != expected:
                raise ValueError(
                    f'keypoint_oks_sigmas has {len(sigmas)} values, but COCO categories define {expected} keypoints')
            coco_eval.params.kpt_oks_sigmas = sigmas
        return coco_eval

    def metric_names(self, iou_type):
        if iou_type == 'pose':
            names = []
            for threshold in self.keypoint_distance_thresholds:
                suffix = f'{threshold:g}px'
                names.extend((f'Precision_{suffix}', f'Recall_{suffix}', f'F1_{suffix}'))
            names.extend(('Visibility_Precision', 'Visibility_Recall', 'Visibility_F1'))
            return tuple(names)
        return COCO_METRIC_NAMES[iou_type]

    def _reset_pose_counts(self):
        self.pose_counts = {
            threshold: {'tp': 0, 'fp': 0, 'fn': 0}
            for threshold in self.keypoint_distance_thresholds
        }
        self.visibility_counts = {'tp': 0, 'fp': 0, 'fn': 0}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = self._make_coco_eval(iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self._reset_pose_counts()


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        if 'keypoints' in self.iou_types:
            self._update_pose_metrics(predictions)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)

                    img_ids, eval_imgs = evaluate(coco_eval)

                    self.eval_imgs[iou_type].append(eval_imgs)


    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
        gathered = all_gather((self.pose_counts, self.visibility_counts))
        self._reset_pose_counts()
        for pose_counts, visibility_counts in gathered:
            for threshold, counts in pose_counts.items():
                for name in ('tp', 'fp', 'fn'):
                    self.pose_counts[threshold][name] += counts[name]
            for name in ('tp', 'fp', 'fn'):
                self.visibility_counts[name] += visibility_counts[name]

    @staticmethod
    def _precision_recall_f1(counts):
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return precision, recall, f1

    def pose_metric_values(self):
        values = []
        for threshold in self.keypoint_distance_thresholds:
            values.extend(self._precision_recall_f1(self.pose_counts[threshold]))
        values.extend(self._precision_recall_f1(self.visibility_counts))
        return values

    def _update_pose_metrics(self, predictions):
        """Accumulate end-to-end keypoint and visibility metrics in pixels."""
        for image_id, prediction in predictions.items():
            gt_annotations = [
                ann for ann in self.coco_gt.imgToAnns.get(image_id, [])
                if not ann.get('iscrowd', 0) and ann.get('_keypoint_valid', False)
            ]
            keep = prediction['scores'].detach().cpu().numpy() >= self.pose_detection_score_thr
            pred_boxes = prediction['boxes'].detach().cpu().numpy()[keep]
            pred_labels = prediction['labels'].detach().cpu().numpy()[keep]
            pred_keypoints = prediction['keypoints'].detach().cpu().numpy()[keep]
            pred_scores = prediction.get('keypoint_scores')
            if pred_scores is None:
                pred_scores = np.ones(pred_keypoints.shape[:2], dtype=np.float32)
            else:
                pred_scores = pred_scores.detach().cpu().numpy()[keep]

            unmatched_predictions = set(range(len(pred_boxes)))
            for annotation in gt_annotations:
                gt_box = np.asarray(annotation['bbox'], dtype=np.float32).copy()
                gt_box[2:] += gt_box[:2]
                candidates = [
                    index for index in unmatched_predictions
                    if int(pred_labels[index]) == int(annotation['category_id'])
                ]
                pred_index = None
                if candidates:
                    ious = np.asarray([_box_iou_xyxy(pred_boxes[index], gt_box) for index in candidates])
                    best_position = int(ious.argmax())
                    if ious[best_position] >= self.keypoint_match_iou_thr:
                        pred_index = candidates[best_position]

                gt_keypoints = np.asarray(annotation['keypoints'], dtype=np.float32).reshape(-1, 3)
                labelled = gt_keypoints[:, 2] > 0
                visible = gt_keypoints[:, 2] > 1
                if pred_index is None:
                    labelled_count = int(labelled.sum())
                    for counts in self.pose_counts.values():
                        counts['fn'] += labelled_count
                    self.visibility_counts['fn'] += int(visible.sum())
                    continue

                unmatched_predictions.remove(pred_index)
                predicted_visible = pred_scores[pred_index] >= self.keypoint_visibility_thr
                distances = np.linalg.norm(pred_keypoints[pred_index] - gt_keypoints[:, :2], axis=1)
                for threshold, counts in self.pose_counts.items():
                    correct = labelled & predicted_visible & (distances <= threshold)
                    counts['tp'] += int(correct.sum())
                    counts['fp'] += int((labelled & predicted_visible & ~correct).sum())
                    counts['fn'] += int((labelled & ~correct).sum())

                self.visibility_counts['tp'] += int((visible & predicted_visible).sum())
                self.visibility_counts['fp'] += int((~visible & predicted_visible).sum())
                self.visibility_counts['fn'] += int((visible & ~predicted_visible).sum())

            for pred_index in unmatched_predictions:
                predicted_count = int((pred_scores[pred_index] >= self.keypoint_visibility_thr).sum())
                for counts in self.pose_counts.values():
                    counts['fp'] += predicted_count
                self.visibility_counts['fp'] += predicted_count

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask.cpu()[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")


            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            bbox_scores = prediction["scores"]
            labels = prediction["labels"].tolist()
            keypoint_xy = prediction["keypoints"]
            keypoint_scores = prediction.get("keypoint_scores")
            if keypoint_scores is None:
                keypoint_scores = torch.ones(
                    keypoint_xy.shape[:-1], dtype=keypoint_xy.dtype, device=keypoint_xy.device)
            if keypoint_scores.shape != keypoint_xy.shape[:-1]:
                raise ValueError(
                    f'keypoint_scores shape {tuple(keypoint_scores.shape)} does not match '
                    f'keypoints shape {tuple(keypoint_xy.shape)}')

            if self.keypoint_score_mode == 'bbox':
                pose_scores = bbox_scores
            elif self.keypoint_score_mode == 'keypoint':
                pose_scores = keypoint_scores.mean(dim=-1)
            else:
                valid_scores = keypoint_scores > self.keypoint_score_thr
                score_sum = (keypoint_scores * valid_scores).sum(dim=-1)
                score_count = valid_scores.sum(dim=-1).clamp(min=1)
                pose_scores = bbox_scores * (score_sum / score_count)

            # COCO detections require flattened (x, y, v) triplets. As in
            # MMPose, retain each model keypoint confidence in the third slot.
            keypoints = torch.cat(
                (keypoint_xy, keypoint_scores.unsqueeze(-1)), dim=-1
            ).flatten(start_dim=1).tolist()
            scores = pose_scores.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results



def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    
def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _box_iou_xyxy(box1, box2):
    top_left = np.maximum(box1[:2], box2[:2])
    bottom_right = np.minimum(box1[2:], box2[2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod()
    area1 = np.maximum(box1[2:] - box1[:2], 0).prod()
    area2 = np.maximum(box2[2:] - box2[:2], 0).prod()
    return float(intersection / max(area1 + area2 - intersection, 1e-12))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def merge(img_ids, eval_imgs):
    """
    img_ids: list[int]
    eval_imgs: list[np.ndarray], each shape [numCats, numAreaRng, numImgs_rank]
    """
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged = {}

    for ids_rank, evals_rank in zip(all_img_ids, all_eval_imgs):
        for i, img_id in enumerate(ids_rank):
            # evals_rank[..., i] is shape [numCats, numAreaRng]
            if img_id not in merged:
                merged[img_id] = evals_rank[..., i]

    merged_img_ids = np.array(list(merged.keys()))
    merged_eval_imgs = np.stack(list(merged.values()), axis=2)

    return merged_img_ids, merged_eval_imgs




#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
