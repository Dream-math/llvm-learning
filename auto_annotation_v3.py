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
.. _tutorial-relay-quick-start:

Quick Start Tutorial for Compiling Deep Learning Models
=======================================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Truman Tian <https://github.com/SiNZeRo>`_

This example shows how to build a neural network with Relay python frontend and
generates a runtime library for Nvidia GPU with TVM.
Notice that you need to build TVM with cuda and llvm enabled.
"""

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tvm_support_list.png
#      :align: center
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.

import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

import os

from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import run_opt_pass
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.expr import Call, TupleGetItem
from tvm.contrib import utils
from tvm import runtime

######################################################################
# Define Neural Network in Relay
# ------------------------------
# First, let's define a neural network with relay python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in Relay.
# Parameters are initialized with Xavier initializer.
# Relay also supports other model formats such as MXNet, CoreML, ONNX and
# Tensorflow.
#
# In this tutorial, we assume we will do inference on our device and
# the batch size is set to be 1. Input images are RGB color images of
# size 224 * 224. We can call the
# :py:meth:`tvm.relay.expr.TupleWrapper.astext()` to show the network
# structure.

operator_list = ["nn.softmax"]

class CustomizedAnnotation(ExprMutator):
    def __init__(self, compiler):
        self.compiler = compiler
        super().__init__()

    def visit_call(self, expr):
        new_args = []
        new_fn = self.visit(expr.op)
        for arg in expr.args:
            if isinstance(arg, Call):
                if arg.op.name in operator_list:
                    arg = compiler_end(self.visit(arg), self.compiler)
                else:
                    arg = compiler_end(self.visit(arg), "dnnl")

                if expr.op.name in operator_list:
                    arg = compiler_begin(arg, self.compiler)
                else:
                    arg = compiler_begin(arg, "dnnl")
            elif isinstance(arg, TupleGetItem):
                if isinstance(arg.tuple_value, Call):
                    tuple_value = self.visit(arg.tuple_value)
                    if arg.tuple_value.op.name in operator_list:
                        tuple_value = compiler_end(tuple_value, self.compiler)
                        tuple_value = compiler_begin(tuple_value, self.compiler)
                        arg = TupleGetItem(tuple_value, arg.index)
                        arg = compiler_end(arg, self.compiler)
                        arg = compiler_begin(arg, self.compiler)
                    else:
                        tuple_value = compiler_end(tuple_value, "dnnl")
                        tuple_value = compiler_begin(tuple_value, "dnnl")
                        arg = TupleGetItem(tuple_value, arg.index)
                        arg = compiler_end(arg, "dnnl")
                        arg = compiler_begin(arg, "dnnl")
                else:
                    raise Exception("warning unhandled case: {0}".format(type(arg.tuple_value)))
            else:
                if expr.op.name in operator_list:
                    arg = compiler_begin(arg, self.compiler)
                else:
                    arg = compiler_begin(arg, "dnnl")
            new_args.append(arg)
        return Call(new_fn, new_args, expr.attrs, expr.type_args, expr.span)

    # def visit_tuple_getitem(self, op):
    #     if isinstance(op.tuple_value, Call):
    #     tuple_value = self.visit(op.tuple_value)
    #     if not tuple_value.same_as(op.tuple_value):
    #         return TupleGetItem(tuple_value, op.index)
    #     return op


def annotate(anno, fn):
    new_fn = anno.visit(fn)
    new_params = [x for x in new_fn.params]
    # add compiler_end for the last operator in fn
    if new_fn.body.op.name in operator_list:
        new_body = compiler_end(new_fn.body, anno.compiler)
    else:
        new_body = compiler_end(new_fn.body, "dnnl")
    return relay.Function(list(new_params), new_body, new_fn.ret_type, new_fn.type_params, new_fn.attrs)

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)

func = mod["main"]
print(func)
global_var = mod.get_global_var("main")
anno = CustomizedAnnotation("default")
func = annotate(anno, func)
# print(func)
func = run_opt_pass(func, relay.transform.MergeCompilerRegions())
mod.update_func(global_var, func)
# print(func)
# mod = relay.transform.InferType()(mod)
mod = transform.PartitionGraph()(mod)

# partition_func = mod["main"]
# print(partition_func)
# file = open("out.txt", "w")
global_var_set = mod.get_global_vars()
for var in global_var_set:
    # print(var, file = file)
    # print(mod[var], file = file)
    print(var)
    print(mod[var])
# file.close()
