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

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


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


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pickle_result=True):
    """train a VoxelNet model specified by a config file.
    """
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    class_names = list(input_cfg.class_names)
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    #########################
    # Build Target Assigner
    #########################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    ######################
    # Build NetWork
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    # net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    # net_train = torch.nn.DataParallel(net).cuda()
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    ######################
    # Build Optimizer
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build(optimizer_cfg, net.parameters())
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir, [mixed_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, gstep)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # Prepare Input
    ######################

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    data_iter = iter(dataloader)

    ######################
    # Training
    ######################
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    logf.write(proto_str)
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # total_loop = remain_steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()
    try:
        for _ in range(total_loop):
            if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
                steps = train_cfg.steps % train_cfg.steps_per_eval
            else:
                steps = train_cfg.steps_per_eval
            for step in range(steps):
                lr_scheduler.step()
                try:
                    example = next(data_iter)
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                example_tuple = list(example_torch.values())
                example_tuple[11] = torch.from_numpy(example_tuple[11])
                example_tuple[12] = torch.from_numpy(example_tuple[12])

                assert 13 == len(example_tuple), "something write with training input size!"

                # ret_dict = net(example_torch)

                # Training Input form example
                pillar_x = example_tuple[0][:,:,0].unsqueeze(0).unsqueeze(0)
                pillar_y = example_tuple[0][:,:,1].unsqueeze(0).unsqueeze(0)
                pillar_z = example_tuple[0][:,:,2].unsqueeze(0).unsqueeze(0)
                pillar_i = example_tuple[0][:,:,3].unsqueeze(0).unsqueeze(0)
                num_points_per_pillar = example_tuple[1].float().unsqueeze(0)

                ################################################################
                # Find distance of x, y, z from pillar center
                # assume config_file xyres_16.proto
                coors_x = example_tuple[2][:, 3].float()
                coors_y = example_tuple[2][:, 2].float()
                # self.x_offset = self.vx / 2 + pc_range[0]
                # self.y_offset = self.vy / 2 + pc_range[1]
                # this assumes xyres 20
                # x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
                # y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
                ################################################################

                # assumes xyres_16
                x_sub = coors_x.unsqueeze(1) * 0.16 + 0.08
                y_sub = coors_y.unsqueeze(1) * 0.16 - 39.6
                ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
                x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
                y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

                num_points_for_a_pillar = pillar_x.size()[3]
                mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
                mask = mask.permute(0, 2, 1)
                mask = mask.unsqueeze(1)
                mask = mask.type_as(pillar_x)

                coors = example_tuple[2]
                anchors = example_tuple[6]
                labels = example_tuple[8]
                reg_targets = example_tuple[9]

                input = [pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar,
                         x_sub_shaped, y_sub_shaped, mask, coors, anchors, labels, reg_targets]

                ret_dict = net(input)

                assert 10 == len(ret_dict), "something write with training output size!"

                cls_preds = ret_dict[5]
                loss = ret_dict[0].mean()
                cls_loss_reduced = ret_dict[7].mean()
                loc_loss_reduced = ret_dict[8].mean()
                cls_pos_loss = ret_dict[3]
                cls_neg_loss = ret_dict[4]
                loc_loss = ret_dict[2]
                cls_loss = ret_dict[1]
                dir_loss_reduced = ret_dict[6]
                cared = ret_dict[9]
                labels = example_tuple[8]
                if train_cfg.enable_mixed_precision:
                    loss *= loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                mixed_optimizer.step()
                mixed_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                # if 'anchors_mask' not in example_torch:
                #     num_anchors = example_torch['anchors'].shape[1]
                # else:
                #     num_anchors = int(example_torch['anchors_mask'][0].sum())
                num_anchors = int(example_tuple[7][0].sum())
                global_step = net.get_global_step()
                if global_step % display_step == 0:
                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    metrics.update(net_metrics)
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(cls_neg_loss.detach().cpu().numpy())
                    # if unlabeled_training:
                    #     metrics["loss"]["diff_rt"] = float(
                    #         diff_loc_loss_reduced.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(dir_loss_reduced.detach().cpu().numpy())


                    metrics["num_vox"] = int(example_tuple[0].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    metrics["lr"] = float(mixed_optimizer.param_groups[0]['lr'])
                    metrics["image_idx"] = example_tuple[11][0]
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    for k, v in flatted_summarys.items():
                        if isinstance(v, (list, tuple)):
                            v = {str(i): e for i, e in enumerate(v)}
                            writer.add_scalars(k, v, global_step)
                        else:
                            writer.add_scalar(k, v, global_step)
                    metrics_str_list = []
                    for k, v in flatted_metrics.items():
                        if isinstance(v, float):
                            metrics_str_list.append(f"{k}={v:.3}")
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f"{e:.3}" for e in v])
                                metrics_str_list.append(f"{k}=[{v_str}]")
                            else:
                                metrics_str_list.append(f"{k}={v}")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    log_str = ', '.join(metrics_str_list)
                    print(log_str, file=logf)
                    print(log_str)
                ckpt_elasped_time = time.time() - ckpt_start_time
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())
                    ckpt_start_time = time.time()

            total_step_elapsed += steps
            torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())

            # Ensure that all evaluation points are saved forever
            torchplus.train.save_models(eval_checkpoint_dir, [net, optimizer], net.get_global_step(), max_to_keep=100)

    except Exception as e:
        torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())
        logf.close()
        raise e
    # save model before exit
    torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())
    logf.close()


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

    # eval example : [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
    #                 4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
    #                 8: 'image_idx', 9: 'image_shape']


    batch_image_shape = example[9]

    batch_imgidx = example[8]

    pillar_x = example[0][:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = example[0][:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = example[0][:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = example[0][:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example[1].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example[2][:, 3].float()
    coors_y = example[2][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors   = example[2]
    anchors = example[6]
    anchors_mask = example[7]
    anchors_mask = torch.as_tensor(anchors_mask, dtype=torch.uint8, device=pillar_x.device)
    anchors_mask = anchors_mask.byte()
    rect = example[3]
    Trv2c = example[4]
    P2 = example[5]
    image_idx = example[8]

    input = [pillar_x, pillar_y, pillar_z, pillar_i,
             num_points_per_pillar, x_sub_shaped, y_sub_shaped,
             mask, coors, anchors, anchors_mask, rect, Trv2c, P2, image_idx]

    predictions_dicts = net(input)

    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict[5]

        if preds_dict[0] is not None: # bbox list
            box_2d_preds = preds_dict[0].detach().cpu().numpy() # bbox
            box_preds = preds_dict[1].detach().cpu().numpy() # bbox3d_camera
            scores = preds_dict[3].detach().cpu().numpy() # scores
            box_preds_lidar = preds_dict[2].detach().cpu().numpy() # box3d_lidar
            # write pred to file
            label_preds = preds_dict[4].detach().cpu().numpy() # label_preds

            anno = kitti.get_start_result_anno()
            num_example = 0
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
                image_shape = [image_shape[0], image_shape[1]]
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


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

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

    for example in iter(eval_dataloader):
        # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
        #               4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
        #               8: 'image_idx', 9: 'image_shape']
        example = example_convert_to_torch(example, float_dtype)

        example_tuple = list(example.values())
        example_tuple[8] = torch.from_numpy(example_tuple[8])
        example_tuple[9] = torch.from_numpy(example_tuple[9])

        if (example_tuple[6].size()[0] != input_cfg.batch_size):
            continue

        if pickle_result:
            dt_annos += predict_kitti_to_anno(
                net, example_tuple, class_names, center_limit_range,
                model_cfg.lidar_input, global_set)
        else:
            _predict_kitti_to_file(net, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
        bar.print_bar()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

    print(f"avg forward time per example: {net.avg_forward_time:.3f}")
    print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        if (len(gt_annos)%2 != 0):
            del gt_annos[-1]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


def export_onnx(net, example, class_names, batch_image_shape,
                center_limit_range=None, lidar_input=False, global_set=None):

    pillar_x = example[0][:,:,0].unsqueeze(0).unsqueeze(0)
    pillar_y = example[0][:,:,1].unsqueeze(0).unsqueeze(0)
    pillar_z = example[0][:,:,2].unsqueeze(0).unsqueeze(0)
    pillar_i = example[0][:,:,3].unsqueeze(0).unsqueeze(0)
    num_points_per_pillar = example[1].float().unsqueeze(0)

    # Find distance of x, y, and z from pillar center
    # assuming xyres_16.proto
    coors_x = example[2][:, 3].float()
    coors_y = example[2][:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100],dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    num_points_for_a_pillar = pillar_x.size()[3]
    mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
    mask = mask.permute(0, 2, 1)
    mask = mask.unsqueeze(1)
    mask = mask.type_as(pillar_x)

    coors = example[2]

    print(pillar_x.size())
    print(pillar_y.size())
    print(pillar_z.size())
    print(pillar_i.size())
    print(num_points_per_pillar.size())
    print(x_sub_shaped.size())
    print(y_sub_shaped.size())
    print(mask.size())

    input_names = ["pillar_x", "pillar_y", "pillar_z", "pillar_i",
                   "num_points_per_pillar", "x_sub_shaped", "y_sub_shaped", "mask"]

    # Wierd Convloution
    pillar_x = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_y = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_z = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    pillar_i = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    num_points_per_pillar = torch.ones([1, 12000], dtype=torch.float32, device=pillar_x.device)
    x_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    y_sub_shaped = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)
    mask = torch.ones([1, 1, 12000, 100], dtype=torch.float32, device=pillar_x.device)

    # De-Convolution
    # pillar_x = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_y = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_z = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # pillar_i = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )
    # num_points_per_pillar = torch.ones([1, 8599],dtype=torch.float32, device=pillar_x.device )
    # x_sub_shaped = torch.ones([1, 100,8599, 1],dtype=torch.float32, device=pillar_x.device )
    # y_sub_shaped = torch.ones([1, 100,8599, 1],dtype=torch.float32, device=pillar_x.device )
    # mask = torch.ones([1, 100, 8599, 1],dtype=torch.float32, device=pillar_x.device )

    example1 = [pillar_x, pillar_y, pillar_z, pillar_i,
                num_points_per_pillar, x_sub_shaped, y_sub_shaped, mask]

    print('-------------- network readable visiual --------------')
    torch.onnx.export(net, example1, "pfe.onnx", verbose=False, input_names=input_names)
    print('pfe.onnx transfer success ...')

    rpn_input = torch.ones([1, 64, 496, 432], dtype=torch.float32, device=pillar_x.device)
    torch.onnx.export(net.rpn, rpn_input, "rpn.onnx", verbose=False)
    print('rpn.onnx transfer success ...')

    return 0


def onnx_model_generate(config_path,
                        model_dir,
                        result_path=None,
                        predict_test=False,
                        ckpt_path=None
                        ):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range

    ##########################
    ## Build Voxel Generator
    ##########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)


    net = second_builder.build(model_cfg, voxel_generator, target_assigner, 1)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=merge_second_batch)


    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)

    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start(len(eval_dataset) // input_cfg.batch_size + 1)

    for example in iter(eval_dataloader):
        example = example_convert_to_torch(example, float_dtype)
        example_tuple = list(example.values())
        batch_image_shape = example_tuple[8]
        example_tuple[8] = torch.from_numpy(example_tuple[8])
        example_tuple[9] = torch.from_numpy(example_tuple[9])

        dt_annos = export_onnx(
            net, example_tuple, class_names, batch_image_shape, center_limit_range,
            model_cfg.lidar_input, global_set)
        return 0
        bar.print_bar()

if __name__ == '__main__':
    fire.Fire()
