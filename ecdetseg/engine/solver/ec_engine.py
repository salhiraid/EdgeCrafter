"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import math
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler = kwargs.get('lr_warmup_scheduler', None)
    output_dir = kwargs.get('output_dir', None)
    train_gt_visualization_interval = int(
        kwargs.get('train_gt_visualization_interval', 0) or 0)
    train_gt_visualization_images = int(
        kwargs.get('train_gt_visualization_images', 1) or 1)
    matcher_debug_interval = int(kwargs.get('matcher_debug_interval', 0) or 0)
    matcher_debug_images = int(kwargs.get('matcher_debug_images', 2) or 2)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _validate_target_labels(targets, criterion.num_classes)
        _validate_target_boxes(targets)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if (train_gt_visualization_interval > 0
                and global_step % train_gt_visualization_interval == 0
                and dist_utils.is_main_process()):
            _visualize_training_ground_truth(
                samples, targets, writer, output_dir, epoch, global_step,
                max_images=train_gt_visualization_images)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if (matcher_debug_interval > 0
                and global_step % matcher_debug_interval == 0
                and dist_utils.is_main_process()):
            _debug_hungarian_matches(
                samples, targets, outputs, criterion.matcher, writer,
                output_dir, epoch, global_step, matcher_debug_images)

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                group_name = pg.get('name', f'pg_{j}')
                writer.add_scalar(f'Lr/{group_name}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _validate_target_labels(targets, num_classes):
    """Fail on CPU with context instead of an asynchronous CUDA gather assert."""
    for batch_index, target in enumerate(targets):
        labels = target.get('labels')
        if labels is None or labels.numel() == 0:
            continue
        invalid = (labels < 0) | (labels >= num_classes)
        if invalid.any():
            invalid_labels = sorted(set(labels[invalid].detach().cpu().tolist()))
            dataset_index = target.get('dataset_index')
            dataset_index = int(dataset_index.item()) if dataset_index is not None else 'unknown'
            image_id = target.get('image_id')
            image_id = int(image_id.item()) if image_id is not None else 'unknown'
            raise ValueError(
                f'Target labels must be in [0, {num_classes - 1}], but found {invalid_labels} '
                f'at batch_index={batch_index}, dataset_index={dataset_index}, image_id={image_id}. '
                'Use a shared contiguous category-name mapping for every component dataset.')


def _validate_target_boxes(targets):
    """Validate the normalized CXCYWH contract expected by box losses."""
    for batch_index, target in enumerate(targets):
        boxes = target.get('boxes')
        if boxes is None or boxes.numel() == 0:
            continue
        dataset_index = target.get('dataset_index')
        dataset_index = int(dataset_index.item()) if dataset_index is not None else 'unknown'
        image_id = target.get('image_id')
        image_id = int(image_id.item()) if image_id is not None else 'unknown'
        context = (
            f'batch_index={batch_index}, dataset_index={dataset_index}, '
            f'image_id={image_id}')
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(
                f'Target boxes must have shape [N, 4], got {tuple(boxes.shape)} at {context}')
        if not torch.isfinite(boxes).all():
            raise ValueError(f'Target boxes contain NaN or Inf at {context}')
        box_min = float(boxes.min().detach().cpu())
        box_max = float(boxes.max().detach().cpu())
        if box_min < -1e-4 or box_max > 1.0001:
            raise ValueError(
                'Target boxes must be normalized CXCYWH values in [0, 1], '
                f'but their range is [{box_min:.4f}, {box_max:.4f}] at {context}. '
                'Ensure ConvertBoxes(fmt="cxcywh", normalize=True) receives a '
                'torchvision BoundingBoxes object after sanitization.')
        if (boxes[:, 2:] <= 0).any():
            raise ValueError(f'Target boxes contain non-positive width/height at {context}')


def _visualize_training_ground_truth(samples, targets, writer, output_dir,
                                     epoch, global_step, max_images=1):
    """Render post-augmentation GT boxes/keypoints for pipeline inspection."""
    if len(samples) == 0 or max_images <= 0:
        return 0

    mean = samples.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = samples.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(global_step))
    pose_candidates = []
    for batch_index, target in enumerate(targets):
        keypoints = target.get('keypoints')
        keypoint_valid = target.get(
            'keypoint_valid', target.get('has_keypoints'))
        has_annotated_pose = (
            keypoints is not None and keypoints.numel() > 0
            and (keypoints[..., 2] > 0).any()
            and (keypoint_valid is None or keypoint_valid.bool().any()))
        if has_annotated_pose:
            pose_candidates.append(batch_index)
    candidates = pose_candidates or list(range(len(samples)))
    order = torch.randperm(len(candidates), generator=generator)[:max_images]
    selected = [candidates[index] for index in order.tolist()]

    save_dir = None
    if output_dir is not None:
        save_dir = (Path(output_dir) / 'training_ground_truth'
                    / f'epoch_{int(epoch):04d}')
        save_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for slot, batch_index in enumerate(selected):
        sample = samples[batch_index]
        target = targets[batch_index]
        image_tensor = (
            sample.detach().cpu() * std.detach().cpu() + mean.detach().cpu()
        ).clamp(0, 1)
        image_array = (
            image_tensor.mul(255).byte().permute(1, 2, 0).numpy())
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        boxes = target.get('boxes')
        labels = target.get('labels')
        if boxes is not None:
            boxes = boxes.detach().cpu()
            labels = labels.detach().cpu() if labels is not None else None
            for instance_index, (cx, cy, bw, bh) in enumerate(boxes.tolist()):
                x1 = (cx - bw / 2.0) * width
                y1 = (cy - bh / 2.0) * height
                x2 = (cx + bw / 2.0) * width
                y2 = (cy + bh / 2.0) * height
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                label = int(labels[instance_index]) if labels is not None else -1
                draw.text((x1 + 2, y1 + 2), f'gt:{label}', fill=(0, 255, 0))

        keypoints = target.get('keypoints')
        keypoint_valid = target.get(
            'keypoint_valid', target.get('has_keypoints'))
        if keypoints is not None:
            keypoints = keypoints.detach().cpu()
            if keypoint_valid is None:
                keypoint_valid = torch.ones(
                    len(keypoints), dtype=torch.bool)
            else:
                keypoint_valid = keypoint_valid.detach().cpu().bool()
            for instance_index, instance_keypoints in enumerate(keypoints):
                if not bool(keypoint_valid[instance_index]):
                    continue
                for keypoint_index, (x, y, visibility) in enumerate(
                        instance_keypoints.tolist()):
                    if visibility <= 0:
                        continue
                    px, py = x * width, y * height
                    if not (0 <= px < width and 0 <= py < height):
                        continue
                    color = (255, 64, 64) if visibility >= 2 else (255, 165, 0)
                    radius = 3
                    draw.ellipse(
                        (px - radius, py - radius, px + radius, py + radius),
                        fill=color, outline=(255, 255, 255))
                    draw.text((px + 4, py - 4), str(keypoint_index), fill=color)

        image_id = target.get('image_id')
        image_id = int(image_id.item()) if image_id is not None else batch_index
        if save_dir is not None:
            image.save(
                save_dir / f'step_{int(global_step):08d}_image_{image_id}.jpg',
                quality=95)
        if writer is not None:
            tensorboard_image = torch.from_numpy(
                np.asarray(image).copy()).permute(2, 0, 1)
            writer.add_image(
                f'Training_ground_truth/sample_{slot}', tensorboard_image,
                global_step)
        rendered += 1
    return rendered


@torch.no_grad()
def _debug_hungarian_matches(samples, targets, outputs, matcher, writer,
                             output_dir, epoch, global_step, max_images=2):
    """Save matched GT/prediction overlays and per-pair cost breakdowns."""
    main_outputs = {
        key: value for key, value in outputs.items()
        if 'aux' not in key and key not in ('dn_outputs', 'dn_pre_outputs')}
    match_result = matcher(
        main_outputs, targets, return_diagnostics=True)
    indices = match_result['indices']
    diagnostics = match_result['diagnostics']
    probabilities = (main_outputs['pred_logits'].sigmoid()
                     if matcher.use_focal_loss else
                     main_outputs['pred_logits'].softmax(-1))

    root = Path(output_dir or '.') / 'matcher_debug'
    image_dir = root / f'epoch_{int(epoch):04d}'
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / 'matches.jsonl'
    mean = samples.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = samples.new_tensor([0.229, 0.224, 0.225])[:, None, None]

    records = []
    for batch_index in range(min(len(samples), max_images)):
        image_tensor = (
            samples[batch_index].detach().cpu() * std.detach().cpu()
            + mean.detach().cpu()).clamp(0, 1)
        image = Image.fromarray(
            image_tensor.mul(255).byte().permute(1, 2, 0).numpy())
        draw = ImageDraw.Draw(image)
        width, height = image.size
        pred_indices, target_indices = indices[batch_index]
        target = targets[batch_index]
        diag = diagnostics[batch_index]

        for pair_index, (pred_index, target_index) in enumerate(zip(
                pred_indices.tolist(), target_indices.tolist())):
            gt_box = target['boxes'][target_index].detach().cpu().tolist()
            pred_box = main_outputs['pred_boxes'][batch_index, pred_index].detach().cpu().tolist()
            label = int(target['labels'][target_index])
            score = float(probabilities[batch_index, pred_index, label].detach().cpu())
            pose_annotation_valid = bool(target.get(
                'keypoint_valid', target.get('has_keypoints', torch.ones(
                    len(target['boxes']), device=target['boxes'].device,
                    dtype=torch.bool)))[target_index])
            gt_keypoints = target.get('keypoints')
            pose_coordinate_valid = bool(
                pose_annotation_valid and gt_keypoints is not None
                and (gt_keypoints[target_index, :, 2] > 0).any())
            component_costs = {
                name: float(matrix[pred_index, target_index])
                for name, matrix in diag['components'].items()}
            target_costs = diag['total_cost'][:, target_index]
            alternative_queries = torch.argsort(target_costs)[:5].tolist()
            record = {
                'step': int(global_step), 'epoch': int(epoch),
                'batch_index': batch_index,
                'image_id': int(target['image_id'].item()),
                'pair_index': pair_index, 'query_index': pred_index,
                'target_index': target_index, 'label': label,
                'predicted_label': int(probabilities[batch_index, pred_index].argmax()),
                'class_score': score,
                'pose_annotation_valid': pose_annotation_valid,
                'pose_coordinate_valid': pose_coordinate_valid,
                'total_cost': float(diag['total_cost'][pred_index, target_index]),
                'weighted_costs': component_costs,
                'top_query_alternatives': [
                    {'query_index': query_index,
                     'total_cost': float(target_costs[query_index])}
                    for query_index in alternative_queries],
                'gt_box_cxcywh': gt_box, 'pred_box_cxcywh': pred_box,
            }
            records.append(record)

            for box, color in ((gt_box, (0, 255, 0)), (pred_box, (0, 128, 255))):
                cx, cy, bw, bh = box
                xyxy = ((cx - bw / 2) * width, (cy - bh / 2) * height,
                        (cx + bw / 2) * width, (cy + bh / 2) * height)
                draw.rectangle(xyxy, outline=color, width=2)
            gx, gy, _, _ = gt_box
            draw.text((gx * width, gy * height),
                      f'GT{target_index}<-Q{pred_index} s={score:.2f} '
                      f'C={record["total_cost"]:.2f}', fill=(255, 255, 0))

            pred_keypoints = main_outputs.get('pred_keypoints')
            pred_visibility = main_outputs.get('pred_keypoint_logits')
            if pose_coordinate_valid and pred_keypoints is not None:
                gt_kpts = gt_keypoints[target_index].detach().cpu()
                pred_kpts = pred_keypoints[batch_index, pred_index].detach().cpu()
                pred_vis_scores = (pred_visibility[batch_index, pred_index]
                                   .detach().float().sigmoid().cpu()
                                   if pred_visibility is not None else None)
                valid_joints = gt_kpts[:, 2] > 0
                errors = torch.linalg.vector_norm(
                    pred_kpts[valid_joints] - gt_kpts[valid_joints, :2], dim=-1)
                record['normalized_keypoint_error_mean'] = float(errors.mean())
                record['normalized_keypoint_error_max'] = float(errors.max())
                record['gt_keypoints_xyv'] = gt_kpts.tolist()
                record['pred_keypoints_xy'] = pred_kpts.tolist()
                record['pred_keypoint_coordinate_std'] = (
                    pred_kpts.float().std(dim=0).tolist())
                if pred_vis_scores is not None:
                    record['predicted_visibility_mean'] = float(
                        pred_vis_scores.mean())
                    record['pred_keypoint_visibility_scores'] = (
                        pred_vis_scores.tolist())

                # Draw every predicted joint, including predictions whose GT
                # joint is unlabeled. A red cross is the actual tensor passed
                # to HungarianMatcher; green circles are GT. This makes a
                # collapsed center prediction immediately distinguishable
                # from a rendering or target-coordinate problem.
                for joint_index, (pred_x, pred_y) in enumerate(pred_kpts.tolist()):
                    px, py = pred_x * width, pred_y * height
                    draw.line((px - 4, py, px + 4, py), fill=(255, 32, 32), width=2)
                    draw.line((px, py - 4, px, py + 4), fill=(255, 32, 32), width=2)
                    draw.text((px + 5, py + 2), f'P{joint_index}', fill=(255, 32, 32))

                for joint_index, ((gt_x, gt_y, visibility), (pred_x, pred_y)) in enumerate(
                        zip(gt_kpts.tolist(), pred_kpts.tolist())):
                    if visibility <= 0:
                        continue
                    draw.ellipse((gt_x * width - 3, gt_y * height - 3,
                                  gt_x * width + 3, gt_y * height + 3),
                                 fill=(0, 255, 0))
                    draw.text((gt_x * width + 4, gt_y * height - 5),
                              f'G{joint_index}', fill=(0, 255, 0))
                    draw.line((gt_x * width, gt_y * height,
                               pred_x * width, pred_y * height),
                              fill=(255, 255, 0), width=1)

        image_id = int(target['image_id'].item())
        image.save(image_dir / f'step_{global_step:08d}_image_{image_id}.jpg', quality=95)
        if writer is not None:
            writer.add_image(
                f'Matcher_debug/sample_{batch_index}',
                torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1),
                global_step)

    with log_path.open('a', encoding='utf-8') as log_file:
        for record in records:
            log_file.write(json.dumps(record) + '\n')


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor,
             data_loader, coco_evaluator: CocoEvaluator, device, writer=None,
             output_dir=None, epoch=0, max_visualizations=10):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    visualized = 0
    validation_image_root = getattr(data_loader.dataset, 'img_folder', None)
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        if visualized < max_visualizations and dist_utils.is_main_process():
            visualized += _visualize_predictions(
                samples, targets, results, writer, output_dir, epoch,
                max_visualizations - visualized,
                coco_evaluator.keypoint_visibility_thr,
                coco_evaluator.pose_detection_score_thr,
                validation_image_root, coco_evaluator.coco_gt)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if 'keypoints' in coco_evaluator.iou_types:
            print("\nPixel-distance and visibility metrics:")
            for name, value in zip(
                    coco_evaluator.metric_names('pose'), coco_evaluator.pose_metric_values()):
                print(f"  {name}: {value:.4f}")

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator.labels is not None:
        
        import numpy as np
        from tabulate import tabulate
        
        res_per_type = {}
        headers = ['class']

        for iou_type in coco_evaluator.iou_types:
            if iou_type not in coco_evaluator.coco_eval:
                continue
            
            precisions = coco_evaluator.coco_eval[iou_type].eval['precision']
            ap = np.mean(precisions[..., 0, -1], axis=(0, 1)) * 100
            ap_50 = np.mean(precisions[0, :, :, 0, -1], axis=0) * 100
            
            prefix = {'bbox': 'bbox', 'segm': 'segm', 'keypoints': 'keypoint'}[iou_type]
            headers.extend([f'{prefix}-AP', f'{prefix}-AP50'])
            res_per_type[iou_type] = (ap, ap_50)

        # Construct rows by merging metrics for each class
        table_data = []
        for k, name in enumerate(coco_evaluator.labels):
            row = [name]
            for iou_type in coco_evaluator.iou_types:
                if iou_type in res_per_type:
                    ap, ap_50 = res_per_type[iou_type]
                    row.extend([f'{ap[k]:.2f}', f'{ap_50[k]:.2f}'])
            table_data.append(row)

        print(f"\n### Class-wise Evaluation Metrics ###")
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        
    
    if coco_evaluator is not None:
        metric_names = {'bbox': 'coco_eval_bbox', 'segm': 'coco_eval_mask', 'keypoints': 'coco_eval_keypoints'}
        for iou_type in iou_types:
            stats[metric_names[iou_type]] = coco_evaluator.coco_eval[iou_type].stats.tolist()
        if 'keypoints' in iou_types:
            stats['pose_eval'] = coco_evaluator.pose_metric_values()

    return stats, coco_evaluator


def _visualize_predictions(samples, targets, results, writer, output_dir, epoch,
                           limit, keypoint_visibility_thr, detection_score_thr,
                           image_root=None, coco_gt=None):
    """Render bbox/keypoint predictions to TensorBoard and JPEG files."""
    if limit <= 0 or (writer is None and output_dir is None):
        return 0
    mean = samples.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = samples.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    saved = 0
    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir) / 'prediction_visualizations' / f'epoch_{int(epoch):04d}'
        save_dir.mkdir(parents=True, exist_ok=True)

    for sample, target, result in zip(samples, targets, results):
        if saved >= limit:
            break
        image_id = int(target['image_id'].item())
        image = _load_original_validation_image(image_root, coco_gt, image_id)
        if image is None:
            image_tensor = ((sample.detach().cpu() * std.cpu()) + mean.cpu()).clamp(0, 1)
            image_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
        image_array = np.asarray(image)
        draw = ImageDraw.Draw(image)
        input_h, input_w = image_array.shape[:2]
        orig_w, orig_h = target['orig_size'].detach().cpu().tolist()
        scale_x, scale_y = input_w / max(orig_w, 1), input_h / max(orig_h, 1)

        scores = result['scores'].detach().cpu()
        keep = scores >= detection_score_thr
        boxes = result['boxes'].detach().cpu()[keep]
        labels = result['labels'].detach().cpu()[keep]
        keypoints = result.get('keypoints')
        keypoint_scores = result.get('keypoint_scores')
        if keypoints is not None:
            keypoints = keypoints.detach().cpu()[keep]
            if keypoint_scores is None:
                keypoint_scores = torch.ones(keypoints.shape[:2])
            else:
                keypoint_scores = keypoint_scores.detach().cpu()[keep]

        for index, (box, score, label) in enumerate(zip(boxes, scores[keep], labels)):
            x1, y1, x2, y2 = box.tolist()
            box_scaled = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            draw.rectangle(box_scaled, outline=(0, 255, 0), width=2)
            draw.text((box_scaled[0], max(0, box_scaled[1] - 12)),
                      f'{int(label)} {float(score):.2f}', fill=(0, 255, 0))
            if keypoints is not None:
                for keypoint_index, ((x, y), kp_score) in enumerate(zip(
                        keypoints[index].tolist(), keypoint_scores[index].tolist())):
                    px, py = x * scale_x, y * scale_y
                    if not (0 <= px < input_w and 0 <= py < input_h):
                        continue
                    # Always render predicted joints so an untrained/low-score
                    # visibility head cannot make TensorBoard look as if the
                    # keypoint branch is missing. Red means visible according
                    # to the evaluation threshold; orange means below it.
                    confident = kp_score >= keypoint_visibility_thr
                    color = (255, 64, 64) if confident else (255, 165, 0)
                    radius = 3 if confident else 2
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                                 fill=color, outline=(255, 255, 255))
                    draw.text((px + radius + 1, py), str(keypoint_index), fill=color)

        if save_dir is not None:
            image.save(save_dir / f'image_{image_id}.jpg', quality=90)
        if writer is not None:
            writer.add_image(
                f'Validation_predictions/image_{image_id}', np.asarray(image),
                global_step=int(epoch), dataformats='HWC')
        saved += 1
    return saved


def _load_original_validation_image(image_root, coco_gt, image_id):
    """Load an untouched validation image using its COCO file name."""
    if image_root is None or coco_gt is None:
        return None
    image_info = coco_gt.imgs.get(image_id)
    if not image_info or not image_info.get('file_name'):
        return None
    image_path = Path(image_root) / image_info['file_name']
    if not image_path.is_file():
        return None
    return Image.open(image_path).convert('RGB')
