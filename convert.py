import os
import sys

import cv2

import numpy as np
import onnx
import netron
import time


def paddle2onnx():
    os.system("paddle2onnx "
              "--model_dir OCR232M/angle_cls "
              "--model_filename inference.pdmodel "
              "--params_filename inference.pdiparams "
              "--save_file ONNXmobile/angle_cls/model1.onnx "
              "--opset_version 11 "
              "--enable_onnx_checker True")

    os.system("paddle2onnx "
              "--model_dir OCR232M/character_rec "
              "--model_filename inference.pdmodel "
              "--params_filename inference.pdiparams "
              "--save_file ONNXmobile/character_rec/model2.onnx "
              "--opset_version 11 "
              "--enable_onnx_checker True")

    os.system("paddle2onnx "
              "--model_dir OCR232M/detect "
              "--model_filename inference.pdmodel "
              "--params_filename inference.pdiparams "
              "--save_file ONNXmobile/detect/model3.onnx "
              "--opset_version 11 "
              "--enable_onnx_checker True")


root = "ONNXmobile"

modelin        = "model"
modelout     = "out"
model_test  = "test"
modelshow = "show"


def func1(change=False, show=False, toShowOutput=False):
    onnx_file = './{}/angle_cls/{}1.onnx'.format(root, modelin)
    onnx_model = onnx.load(onnx_file)

    ginput = onnx_model.graph.input
    goutput = onnx_model.graph.output
    print(ginput, goutput)

    if toShowOutput:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file)), "./{}/angle_cls/{}1.onnx".format(root, modelshow))
        netron.start("./{}/angle_cls/{}1.onnx".format(root, modelshow))

    if show:
        netron.start(onnx_file)

    if change:
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 48
        onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "?" 

        onnx.checker.check_model(onnx_model)
        print('The model is checked!')
        onnx.save_model(onnx_model, "./{}/angle_cls/{}1.onnx".format(root, modelout))


def func2(change=False, show=False, toShowOutput=False):
    onnx_file = './{}/character_rec/{}2.onnx'.format(root, modelin)
    onnx_model = onnx.load(onnx_file)

    ginput = onnx_model.graph.input
    goutput = onnx_model.graph.output
    print(ginput, goutput)

    if toShowOutput:
        #onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file)), "./{}/character_rec/{}2.onnx".format(root, modelshow))
        netron.start("./{}/character_rec/{}2.onnx".format(root, modelshow))

    if show:
        netron.start(onnx_file)

    if change:
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "?"
        onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "?" 

        onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_param = "?"
        # onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "?" 

        onnx.checker.check_model(onnx_model)
        print('The model is checked!')
        onnx.save_model(onnx_model, "./{}/character_rec/{}2.onnx".format(root, modelout))


def func3(change=False, show=False, toShowOutput=False):
    onnx_file = './{}/detect/{}3.onnx'.format(root, "out")
    onnx_model = onnx.load(onnx_file)

    ginput = onnx_model.graph.input
    goutput = onnx_model.graph.output
    print(ginput, goutput)

    if toShowOutput:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file)), "./{}/detect/{}3.onnx".format(root, modelshow))
        netron.start("./{}/detect/{}3.onnx".format(root, modelshow))

    if show:
        netron.start(onnx_file)

    if change:
        onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 720
        onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 1280 

        onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx_model.graph.output[0].type.tensor_type.shape.dim[2].dim_param = "?"
        onnx_model.graph.output[0].type.tensor_type.shape.dim[3].dim_param = "?" 

        onnx.checker.check_model(onnx_model)
        print('The model is checked!')
        onnx.save_model(onnx_model, "./{}/detect/{}3.onnx".format(root, model_test))


def onnx2trt(aim):
    if aim == 1:
        pass
    if aim == 2:
        os.system("/usr/src/tensorrt/bin/trtexec --onnx=./ONNXmobile/character_rec/out2.onnx --explicitBatch \
                --minShapes='x':1x3x32x4\
                --optShapes='x':1x3x32x100\
                --maxShapes='x':1x3x32x1920\
                --workspace=300\
                --saveEngine=./ONNXmobile/character_rec/rec.engine")
    if aim == 3:
        os.system("/usr/src/tensorrt/bin/trtexec --onnx=./ONNXmobile/detect/out3.onnx --explicitBatch \
                --minShapes='x':1x3x192x192\
                --optShapes='x':1x3x512x512\
                --maxShapes='x':1x3x1920x1920\
                --workspace=1700\
                --saveEngine=./ONNXmobile/detect/detect.engine")


# paddle2onnx()
# func1()
func2(show=False)
# func3(change=False, show=False, toShowOutput=False)

# onnx2trt(2)
