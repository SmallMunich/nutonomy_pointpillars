# -*- coding: utf-8 -*-
import fire
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import sys
sys.path.append('../')
import time
# from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter

def onnx_model_check():

    onnx_model = onnx.load("pfe.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx_model)

def tensorrt_backend_pfe_onnx():

    pillar_x = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_y = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_z = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_i = np.ones([1, 1, 12000, 100], dtype=np.float32)

    num_points_per_pillar = np.ones([1, 12000], dtype=np.float32)
    x_sub_shaped = np.ones([1, 1, 12000, 100], dtype=np.float32)
    y_sub_shaped = np.ones([1, 1, 12000, 100], dtype=np.float32)
    mask = np.ones([1, 1, 12000, 100], dtype=np.float32)

    pfe_inputs = [pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar,
                  x_sub_shaped, y_sub_shaped, mask]

    print("pfe_inputs length is : ", len(pfe_inputs))
    start = time.time()

    pfe_model = onnx.load("pfe.onnx")
    engine = backend.prepare(pfe_model, device="CUDA:0", max_batch_size=1)


    for i in range(1, 1000):
        pfe_outputs = engine.run(pfe_inputs)
    end = time.time()
    print('inference time is : ', (end - start)/1000)
    print(pfe_outputs)


def tensorrt_backend_rpn_onnx():

    rpn_input_features = np.ones([1, 64, 496, 432], dtype=np.float32)

    rpn_start_time = time.time()

    rpn_model = onnx.load("rpn.onnx")
    engine = backend.prepare(rpn_model, device="CUDA:0", max_batch_size=1)

    for i in range(1, 1000):
        rpn_outputs = engine.run(rpn_input_features)

    rpn_end_time = time.time()

    print('rpn inference time is : ', (rpn_end_time - rpn_start_time)/1000)
    print(rpn_outputs)

def tensorrt_backend_pointpillars_onnx(config_path=None):
    import torch
    from second.protos import pipeline_pb2
    from google.protobuf import text_format
    from second.builder import voxel_builder
    from second.pytorch.models.pointpillars import PointPillarsScatter

    ############################# PFE-Layer TensorRT ################################
    pillar_x = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_y = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_z = np.ones([1, 1, 12000, 100], dtype=np.float32)
    pillar_i = np.ones([1, 1, 12000, 100], dtype=np.float32)
    num_points_per_pillar = np.ones([1, 12000], dtype=np.float32)
    x_sub_shaped = np.ones([1, 1, 12000, 100], dtype=np.float32)
    y_sub_shaped = np.ones([1, 1, 12000, 100], dtype=np.float32)
    mask = np.ones([1, 1, 12000, 100], dtype=np.float32)

    pfe_inputs = [pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar,
                  x_sub_shaped, y_sub_shaped, mask]

    pfe_model = onnx.load("pfe.onnx")
    engine = backend.prepare(pfe_model, device="CUDA:0", max_batch_size=1)

    pfe_start_time = time.time()
    pfe_outputs = engine.run(pfe_inputs)
    pfe_end_time = time.time()

    print('inference time is : ', (pfe_end_time - pfe_start_time))

    ###################### PillarScatter Python Coder Transfer #########################
    # numpy --> tensor
    pfe_outs = np.array(pfe_outputs)
    voxel_features_tensor = torch.from_numpy(pfe_outs)

    voxel_features = voxel_features_tensor.squeeze()
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
    batch_size = 1
    mid_feature_extractor = PointPillarsScatter(output_shape,
                                                num_input_features,
                                                batch_size)

    device = torch.device("cuda:0")
    coors_numpy = np.loadtxt('coors.txt', dtype=np.int32)
    coors = torch.from_numpy(coors_numpy)
    coors = coors.to(device).cuda() #CPU Tensor --> GPU Tensor

    voxel_features = voxel_features.to(device).cuda()
    rpn_input_features = mid_feature_extractor(voxel_features, coors)

    ########################### RPN Network TensorRT #################################

    rpn_input_features = rpn_input_features.data.cpu().numpy()

    rpn_model = onnx.load("rpn.onnx")
    engine_rpn = backend.prepare(rpn_model, device="CUDA:0", max_batch_size=1)

    rpn_start_time = time.time()
    rpn_outputs = engine_rpn.run(rpn_input_features)
    rpn_end_time = time.time()

    print('rpn inference time is : ', (rpn_end_time - rpn_start_time))
    print(rpn_outputs)

if __name__ == '__main__':
    fire.Fire()