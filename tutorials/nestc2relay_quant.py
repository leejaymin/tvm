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

out_channels = 16
batch_size = 1

data = relay.var("data", shape=(1, 3, 224, 224), dtype="float32")


golden_weight = np.loadtxt("/home/jemin/development/dataset/qnn_test/data0006.txt",delimiter=",",skiprows=1).reshape(64,3,3,3)
# filter (OHWI -> OIHW)
golden_weight = golden_weight.transpose(0,3,1,2)
weight = relay.const(golden_weight)

data_shape = (batch_size, 3, 224, 224)



quantized_output = relay.qnn.op.quantize(
    data,
    output_scale=4,
    output_zero_point=0,
    axis=1,
    out_dtype="int8"
)

simple_net2 = relay.qnn.op.conv2d(data=quantized_output,
                                  weight=weight,
                                  input_zero_point=0,
                                  kernel_zero_point=0,
                                  input_scale=4,
                                  kernel_scale=0.002,
                                  kernel_size=(3,3),
                                  channels=out_channels
                                  )

ref_func = tvm.IRModule.from_expr(simple_net2)
print(ref_func)
net, params = testing.create_workload(simple_net2)

######################################################################
# Build and run with cuda backend
# -------------------------------
# We build and run this network with cuda backend, as usual.
# By setting the logging level to DEBUG, the result of Relay graph compilation will be dumped as pseudo code.
import logging

logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion

target = "cuda"
lib = relay.build_module.build(net, target, params=params)

dev = tvm.device(target, 0)

data = np.loadtxt("/home/jemin/development/dataset/qnn_test/data0001.txt",delimiter=",",skiprows=1).reshape(1,3,224,224)


#data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_cuda = out.numpy()

print(out_cuda)