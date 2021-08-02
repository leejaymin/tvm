# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Using External Libraries in Relay
=================================
**Author**: `Masahiro Masuda <https://github.com/masahi>`_, `Truman Tian <https://github.com/SiNZeRo>`_

This is a short tutorial on how to use external libraries such as cuDNN, or cuBLAS with Relay.

Relay uses TVM internally to generate target specific code. For example, with cuda backend TVM generates cuda kernels for all layers in the user provided network.
But sometimes it is also helpful to incorporate external libraries developed by various vendors into Relay.
Luckily, TVM has a mechanism to transparently call into these libraries.
For Relay users, all we need to do is just to set a target string appropriately.

Before we can use external libraries from Relay, your TVM needs to be built with libraries you want to use.
For example, to use cuDNN, USE_CUDNN option in `cmake/config.cmake` needs to be enabled, and cuDNN include and library directories need to be specified if necessary.

To begin with, we import Relay and TVM.
"""
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing

######################################################################
# Create a simple network
# -----------------------
# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.

out_channels = 4
batch_size = 1

data = relay.var("data", relay.TensorType((batch_size, 4, 5, 5), dtype="float32"))
weight = relay.var("weight", relay.TensorType((4, 4, 3, 3), dtype="int8"))
bias = relay.var("bias", relay.TensorType((4,), dtype="int32"))

# data_scale = relay.var("data_scale", shape=(), dtype="float32")
data_scale = relay.const(np.array(4).astype("float32"))

#data_zp = relay.var("data_zp", shape=(), dtype="int32")
# weight_scale = relay.var("weight_scale", shape=(), dtype="float32")
# weight_zp = relay.var("weight_zp", shape=(), dtype="int32")
# bias_scale = relay.var("bias_scale", shape=(), dtype="float32")
# bias_zp = relay.var("bias_zp", shape=(), dtype="int32")


simple_net = relay.qnn.op.quantize(data, data_scale, relay.const(0, "int32"))
simple_net = relay.qnn.op.conv2d(simple_net,
                                 weight,
                                 input_zero_point=relay.const(0, "int32"),
                                 kernel_zero_point=relay.const(0, "int32"),
                                 input_scale=relay.const(4.0, "float32"),
                                 kernel_scale=relay.const(1.0, "float32"),
                                 kernel_size=(3, 3),
                                 channels=64,
                                 padding=(0, 0),
                                 strides=(1, 1),
                                 dilation=(1, 1),
                                 data_layout="NCHW",
                                 kernel_layout="OIHW",
                                 out_dtype="int32",
                                 )
# simple_net = relay.nn.conv2d(
#     data=simple_net, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(0, 0), data_layout="NCHW", kernel_layout="OIHW",out_dtype="int32"
# )

simple_net = relay.nn.bias_add(simple_net, bias, axis=1)
simple_net = relay.qnn.op.dequantize(
        simple_net,
        input_scale=relay.const(0.008, "float32"),
        input_zero_point=relay.const(0, "int32")
        )

# simple_net = relay.nn.conv2d(
#     data=simple_net, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(0, 0), data_layout="NCHW", kernel_layout="OIHW"
# )
# simple_net = relay.nn.bias_add(simple_net, bias, axis=1)
#
# simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)


mod = tvm.IRModule.from_expr(simple_net)
print(mod)
mod = relay.transform.InferType()(mod)
print(mod)

weight_data = np.loadtxt("/home/jemin/development/dataset/qnn_quant/kernel.txt",delimiter=",",skiprows=1,dtype="int8").reshape(64,3,3,3)
weight_data = weight_data.transpose(0,3,1,2)

bias_data = np.loadtxt("/home/jemin/development/dataset/qnn_quant/bias.txt",delimiter=",",skiprows=1, dtype="int32")
scale = np.array(4).astype("float32")
zp = np.array(0).astype("int32")
params = {}
# params['scale'] = tvm.nd.array(scale , device=tvm.cpu(0))
# params['zp'] = tvm.nd.array(zp , device=tvm.cpu(0))
params['weight'] = tvm.nd.array(weight_data , device=tvm.cpu(0))
params['bias'] = tvm.nd.array(bias_data , device=tvm.cpu(0))


#mod, params = testing.create_workload(simple_net)
#print(mod.astext(show_meta_data=True))

######################################################################
# Build and run with cuda backend
# -------------------------------
# We build and run this network with cuda backend, as usual.
# By setting the logging level to DEBUG, the result of Relay graph compilation will be dumped as pseudo code.
import logging

logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion

#lib = relay.build(mod, target="llvm", params=None)
#rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device("llvm", 0)))

target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
    #lib = relay.build(mod, target)
#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
data = np.loadtxt("/home/jemin/development/dataset/qnn_quant/input.txt",delimiter=",",skiprows=1,dtype="float32").reshape(1,3,224,224)

module = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device("llvm", 0)))
module.set_input("data", data)
#module.set_input(**{"data": data, "scale": scale, "zp": zp})

module.run()
out_cuda = module.get_output(0).numpy()
out_cuda = np.round(out_cuda)
#out_shape = (batch_size, out_channels, 222, 222)
#out = module.get_output(0, tvm.nd.empty(out_shape))
#out_cuda = out.numpy()
print("inference result:")
print(out_cuda.shape)
print(np.sum(out_cuda))
print(out_cuda[0,0,0:-1,0])
print("golden")
golden = np.loadtxt("/home/jemin/development/dataset/qnn_quant/result.txt",delimiter=",",skiprows=1,dtype="int8").reshape(1,64,222,222)
#golden = golden.transpose(0,3,1,2)
print(golden.shape)
print(np.sum(golden))
print(golden[0,0,0:-1,0])
