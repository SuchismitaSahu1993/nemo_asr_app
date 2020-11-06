# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

import os
from tools.filetools import mkdir_p
from tools.System.config import cfg

##############################################################################
# Tools for model deployment
##############################################################################

def get_onnx_cmd(config, encoder, decoder, onnx_encoder, onnx_decoder):
  """Returns command to export nemo (jasper) model to ONNX (.onnx)
  """
  changes = []
  changes.append("--config " + config)
  changes.append("--nn_encoder " + encoder)
  changes.append("--nn_decoder " + decoder)
  changes.append("--onnx_encoder " + onnx_encoder)
  changes.append("--onnx_decoder " + onnx_decoder)

  mkdir_p(cfg.MODEL.ONNX_MODELS)

  cmd = "python {} {}".format(cfg.MODEL.ONNX_SCRIPT, ' '.join(changes))
  return cmd

def get_onnx_trt_cmd(onnx_path, trt_plan):
  """Returns command to export ONNX model to TensorRT plan (.engine)
  """
  # to do add other params
  mkdir_p(cfg.MODEL.TRT_MODELS)

  cmd = "python {} {} {}".format(cfg.MODEL.TRT_SCRIPT, onnx_path, trt_plan)
  return cmd
