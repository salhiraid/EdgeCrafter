"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def _checkpoint_state(checkpoint):
    if 'ema' in checkpoint:
        return checkpoint['ema']['module']
    if 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _infer_num_classes(state):
    """Infer the trained class count from an ECDet/ECDetPose checkpoint."""
    suffixes = (
        'decoder.enc_score_head.weight',
        'decoder.dec_score_head.0.weight',
    )
    values = {
        int(tensor.shape[0])
        for name, tensor in state.items()
        if any(name.endswith(suffix) for suffix in suffixes)
    }
    if not values:
        raise ValueError(
            'Could not infer num_classes from the checkpoint score heads. '
            'Pass --num-classes explicitly.')
    if len(values) != 1:
        raise ValueError(
            f'Checkpoint score heads disagree about num_classes: {sorted(values)}')
    return values.pop()


def main(args, ):
    """main
    """
    checkpoint = None
    state = None
    num_classes = args.num_classes
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = _checkpoint_state(checkpoint)
        checkpoint_num_classes = _infer_num_classes(state)
        if num_classes is not None and num_classes != checkpoint_num_classes:
            raise ValueError(
                f'--num-classes={num_classes} does not match the checkpoint '
                f'class-head size ({checkpoint_num_classes}).')
        num_classes = checkpoint_num_classes

    cfg_kwargs = {'resume': args.resume}
    if num_classes is not None:
        # Override the example config before cfg.model is constructed. This is
        # required when a generic one-class vehicle example is used to export
        # a checkpoint trained with multiple vehicle classes.
        cfg_kwargs['num_classes'] = num_classes
    cfg = YAMLConfig(args.config, **cfg_kwargs)
    
    task = cfg.yaml_cfg['task']

    if args.resume:
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True

        # NOTE load train mode state -> convert to deploy mode
        try:
            cfg.model.load_state_dict(state)
        except RuntimeError as error:
            raise RuntimeError(
                'Checkpoint architecture does not match the export config. '
                f'The checkpoint uses num_classes={num_classes}; ensure the '
                'config also matches its S/M backbone, decoder dimensions, '
                'number of decoder layers, and keypoint-head settings.\n'
                f'Original load error:\n{error}') from error

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(1, 3, *img_size)
    # eval_spatial_size is [height, width], while the postprocessor contract is
    # [width, height]. This matters for non-square pose exports because boxes
    # and keypoints must use the correct independent x/y scales.
    size = torch.tensor([[img_size[1], img_size[0]]], dtype=torch.float32)
    with torch.no_grad():
        exported_outputs = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'
    output_names = ['labels', 'boxes', 'scores']
    if task == 'segmentation':
        output_names.append('masks')
    elif len(exported_outputs) == 5:
        # ECDetPose deploy output: labels, boxes, scores, keypoints and
        # per-keypoint visibility/confidence scores.
        output_names.extend(['keypoints', 'keypoint_scores'])
    elif len(exported_outputs) != 3:
        raise RuntimeError(
            f'Unsupported deploy output count: {len(exported_outputs)}. '
            'Expected 3 (detection), 4 (segmentation), or 5 (pose).')
    
    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--opset', type=int, default=18,)
    parser.add_argument(
        '--num-classes', type=int,
        help='Model class count. With --resume it is inferred from the checkpoint and this option only validates it.')
    parser.add_argument('--check',  action='store_true')
    parser.add_argument('--simplify',  action='store_true')
    args = parser.parse_args()
    main(args)
