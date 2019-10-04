# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import sys
sys.path.append('../')
from pathlib import Path
import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar

def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.
    :param actual_num:
    :param max_num:
    :param axis:
    :return: [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num : [N, M, 1]
    # tiled_actual_num : [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # title_max_num : [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape : [batch_size, max_num]
    return paddings_indicator


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = ["voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect", "Trv2c", "P2"]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(v, dtype=torch.uint8, device=device)
            # torch.uint8 is now deprecated, please use a dtype torch.bool instead
        else:
            example_torch[k] = v
    return example_torch


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3, 6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):

    predictions_dicts = net(example)

    return predictions_dicts

def generate_example():
    # check this pfe outputs for origin pytorch model 
    # pillar_x = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_y = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_z = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_i = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # num_points_per_pillar = torch.ones([1, 12000], dtype=torch.float32, device="cuda:0")
    # x_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # y_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # mask = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")

    # check this rpn outputs for origin pytorch model 
    pillar_x = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_y = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_z = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_i = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    num_points_per_pillar = torch.ones([1, 9918], dtype=torch.float32, device="cuda:0")
    x_sub_shaped = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    y_sub_shaped = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    mask = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")

    device = torch.device("cuda:0")
    coors_numpy = np.loadtxt('coors.txt', dtype=np.int32)
    coors = torch.from_numpy(coors_numpy)
    coors = coors.to(device)
    # coors = coors.to(device).cuda()
    example = [pillar_x, pillar_y, pillar_z, pillar_i,
                num_points_per_pillar, x_sub_shaped, y_sub_shaped, mask, coors]

    return example


def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):

    model_dir = str(Path(model_dir).resolve())
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    # dt_annos = []
    global_set = None
    print("Generate output labels...")

    example_tuple = generate_example()

    dt_annos = predict_kitti_to_anno(net, example_tuple, class_names, center_limit_range,
                                     model_cfg.lidar_input, global_set)

    print(dt_annos)

def onnx_model_check():
    import onnx

    onnx_model = onnx.load("pde.onnx")
    onnx.checker.check_model(onnx_model)


def onnx_model_predict(config_path=None, model_dir=None):
    import onnxruntime
    from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter

    # check the pfe onnx model IR input paramters as follows 
    # pillar_x = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_y = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_z = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # pillar_i = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # num_points_per_pillar = torch.ones([1, 12000], dtype=torch.float32, device="cuda:0")
    # x_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # y_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")
    # mask = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device="cuda:0")

    # check the rpn onnx model IR input paramters as follows 
    pillar_x = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_y = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_z = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    pillar_i = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    num_points_per_pillar = torch.ones([1, 9918], dtype=torch.float32, device="cuda:0")
    x_sub_shaped = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    y_sub_shaped = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")
    mask = torch.ones([1, 1, 9918, 100], dtype=torch.float32, device="cuda:0")


    pfe_session = onnxruntime.InferenceSession("pfe.onnx")

    # Compute ONNX Runtime output prediction
    pfe_inputs = {pfe_session.get_inputs()[0].name: (pillar_x.data.cpu().numpy()),
                  pfe_session.get_inputs()[1].name: (pillar_y.data.cpu().numpy()),
                  pfe_session.get_inputs()[2].name: (pillar_z.data.cpu().numpy()),
                  pfe_session.get_inputs()[3].name: (pillar_i.data.cpu().numpy()),
                  pfe_session.get_inputs()[4].name: (num_points_per_pillar.data.cpu().numpy()),
                  pfe_session.get_inputs()[5].name: (x_sub_shaped.data.cpu().numpy()),
                  pfe_session.get_inputs()[6].name: (y_sub_shaped.data.cpu().numpy()),
                  pfe_session.get_inputs()[7].name: (mask.data.cpu().numpy())}

    pfe_outs = pfe_session.run(None, pfe_inputs)
    print('-------------------------- PFE ONNX Outputs ----------------------------')
    print(pfe_outs) # also you could save it to file for comparing
    print('-------------------------- PFE ONNX Ending ----------------------------')
    ##########################Middle-Features-Extractor#########################
    # numpy --> tensor
    pfe_outs = np.array(pfe_outs)
    voxel_features_tensor = torch.from_numpy(pfe_outs)

    voxel_features = voxel_features_tensor.squeeze()
    # voxel_features = np.array(pfe_outs).squeeze()
    voxel_features = voxel_features.permute(1, 0)

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
    model_cfg = config.model.second
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    grid_size = voxel_generator.grid_size
    output_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_input_features = vfe_num_filters[-1]
    batch_size = 2
    mid_feature_extractor = PointPillarsScatter(output_shape,
                                                num_input_features,
                                                batch_size)

    device = torch.device("cuda:0")
    coors_numpy = np.loadtxt('./onnx_predict_outputs/coors.txt', dtype=np.int32)
    coors = torch.from_numpy(coors_numpy)
    coors = coors.to(device).cuda() # CPU Tensor --> GPU Tensor

    voxel_features = voxel_features.to(device).cuda()
    rpn_input_features = mid_feature_extractor(voxel_features, coors)

    #################################RPN-Feature-Extractor########################################
    # rpn_input_features = torch.ones([1, 64, 496, 432], dtype=torch.float32, device='cuda:0')
    rpn_session = onnxruntime.InferenceSession("rpn.onnx")
    # compute RPN ONNX Runtime output prediction
    rpn_inputs = {rpn_session.get_inputs()[0].name: (rpn_input_features.data.cpu().numpy())}

    rpn_outs = rpn_session.run(None, rpn_inputs)
    print('---------------------- RPN ONNX Outputs ----------------------')
    print(rpn_outs)
    print('---------------------- RPN ONNX Ending ----------------------')

if __name__ == '__main__':
    fire.Fire()

