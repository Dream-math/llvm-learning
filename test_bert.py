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
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# Tensorflow imports
from pathlib import Path
import tensorflow as tf
from pytorch_transformers.tokenization_bert import BertTokenizer

import os
import random
import numpy as np
import multiprocessing as mp
import json
import collections

# tvm, relay
import tvm
from tvm import te, relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import *


try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor

from utils.bert_utils import read_squad_examples, input_to_squad_example, squad_examples_to_features, get_answer

# import pdb
# print(f'Python: {os.getpid()}')
# pdb.set_trace()

predict_file = "./dev-v1.1.json"
model_path = "./bert_base.pb"


flags = tf_compat_v1.flags
flags.DEFINE_string("vocab_file", "../data/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 384, # 128
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")
flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")
flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


FLAGS = flags.FLAGS

eval_examples = read_squad_examples(predict_file)
tokenizer = BertTokenizer.from_pretrained(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
# print(features[0].input_ids)
# print(features[0].input_mask)
# print(features[0].segment_ids)
calibration_samples = 20

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

def calibrate_dataset():
    
    calib_data = []
    for example in random.sample(eval_examples, calibration_samples):
    # for example in eval_examples[:calibration_samples]:
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
        for feature in features:
            input_ids = np.array([feature.input_ids])
            input_mask = np.array([feature.input_mask])
            segment_ids = np.array([feature.segment_ids])
            calib_data.append({'input_ids_1': input_ids, 'input_mask_1':input_mask, 'segment_ids_1':segment_ids})

    return calib_data

def quantize(mod, params, data_aware):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="percentile", weight_scale="max", skip_conv_layers=[], skip_dense_layer=False, do_simulation=True): #, dtype_input="uint8", debug_enabled_ops=["nn.conv2d"], calibrate_chunk_by=16
            mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def create_graph(model_path):
    print(f'create_graph...')
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        with tf_compat_v1.Session() as sess:
            # graph_def = tf_testing.AddShapesToGraphDef(sess, "unstack")
            graph_def = tf_testing.AddShapesToGraphDef(sess, "bert/encoder/layer_0/attention/output/add")
    shape_dict = {"input_ids_1": (1, 384), "input_mask_1": (1, 384), "segment_ids_1": (1, 384)}
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict) #, outputs=["bert/encoder/layer_0/attention/self/transpose", "bert/encoder/layer_0/attention/self/transpose_1"])

    return mod, params

def run_tf(model_path):
    print(f'create_graph...')
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        doc = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
        q = "Which NFL team represented the AFC at Super Bowl 50?"
        example = input_to_squad_example(doc, q)
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)

        # Add shapes to the graph.
        with tf_compat_v1.Session() as sess:
            out_0, out_1 = sess.run(("unstack:0", "unstack:1"), feed_dict={"input_ids_1:0": np.array([features[0].input_ids]).astype("int32"),
                                                    "input_mask_1:0": np.array([features[0].input_mask]).astype("int32"),
                                                    "segment_ids_1:0": np.array([features[0].segment_ids]).astype("int32")})
            unique_id = int(features[0].unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = out_0[0].tolist(),
                                end_logits   = out_1[0].tolist())   

            print(out_0[0].tolist())
            print(out_1[0].tolist())

            answer = get_answer(example, features, [result], FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
            print(answer['answer']) # "Denver Broncos"


def run_test(lib):
    doc = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    q = "Which NFL team represented the AFC at Super Bowl 50?"
    example = input_to_squad_example(doc, q)
    features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
    print(features[0].input_ids)
    print(features[0].segment_ids)
    print(features[0].input_mask)
    np.savetxt("/home/exec/test_bert1.txt", features[0].input_ids, fmt='%i', delimiter=' ')
    np.savetxt("/home/exec/test_bert2.txt", features[0].segment_ids, fmt='%i', delimiter=' ')
    np.savetxt("/home/exec/test_bert3.txt", features[0].input_mask, fmt='%i', delimiter=' ')
    input_ids = np.array([features[0].input_ids])
    segment_ids = np.array([features[0].segment_ids])
    input_mask = np.array([features[0].input_mask])
    
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input("input_ids_1", tvm.nd.array(input_ids.astype("int32"))) # set inputs
    m.set_input("segment_ids_1", tvm.nd.array(segment_ids.astype("int32"))) # set inputs
    m.set_input("input_mask_1", tvm.nd.array(input_mask.astype("int32"))) # set inputs
    m.run() # execute
    tvm_output_0 = m.get_output(0).asnumpy() # get outputs
    print("tvm_output_0")
    print(tvm_output_0, tvm_output_0.shape)
    # tvm_output_1 = m.get_output(1).asnumpy() # get outputs
    # print("tvm_output_1")
    # print(tvm_output_1, tvm_output_1.shape)

    # unique_id = int(features[0].unique_id)
    # result = RawResult(unique_id    = unique_id,
    #                     start_logits = tvm_output_0[0].tolist(),
    #                     end_logits   = tvm_output_1[0].tolist())   

    # answer = get_answer(example, features, [result], FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
    # print("answer")
    # print(answer['answer']) # "Denver Broncos"

    return tvm_output_0


def run_inference(lib):
    # eval_examples = read_squad_examples(predict_file)
    # tokenizer = BertTokenizer.from_pretrained(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    m = graph_executor.GraphModule(lib["default"](dev))
    all_result = {}
    # print(len(eval_examples))
    for i, example in enumerate(eval_examples): # [:100]
        features = squad_examples_to_features(example, tokenizer, FLAGS.max_seq_length, FLAGS.doc_stride, FLAGS.max_query_length)
        result = []
        for feature in features:
            input_ids = np.array([feature.input_ids])
            input_mask = np.array([feature.input_mask])
            segment_ids = np.array([feature.segment_ids])

            m.set_input("input_ids_1", tvm.nd.array(input_ids.astype("int32"))) # set inputs
            m.set_input("input_mask_1", tvm.nd.array(input_mask.astype("int32"))) # set inputs
            m.set_input("segment_ids_1", tvm.nd.array(segment_ids.astype("int32"))) # set inputs
            m.run() # execute
            tvm_output_0 = m.get_output(0).asnumpy() # get outputs
            tvm_output_1 = m.get_output(1).asnumpy() # get outputs

            unique_id = int(feature.unique_id)
            result.append(RawResult(unique_id    = unique_id,
                                start_logits = tvm_output_0[0].tolist(),
                                end_logits   = tvm_output_1[0].tolist()))

        answer = get_answer(example, features, result, FLAGS.n_best_size, FLAGS.max_answer_length, FLAGS.do_lower_case)
        all_result.update({example.qas_id : answer['answer']})
        # print(answer['answer'])
        print(i)

    json_str = json.dumps(all_result, indent=4)
    with open('predictions.json', 'w') as fp:
        fp.write(json_str)

def embedding():
    _0 = is_op("expand_dims")(wildcard())
    _1 = is_op("reshape")(_0)
    _2 = is_op("take")(is_constant(), _1)
    _3 = is_op("reshape")(wildcard())
    _4 = is_op("one_hot")(_3, is_constant(), is_constant())
    _5 = is_op("nn.dense")(_4, is_constant())
    _6 = is_op("reshape")(_2)
    _7 = is_op("reshape")(_5)
    _8 = is_op("add")(_6, _7)
    _9 = is_op("add")(_8, is_constant())
    _10 = is_op("mean")(_9)
    _11 = is_op("subtract")(_9, _10)
    _12 = is_op("multiply")(_11, _11)
    _13 = is_op("mean")(_12)
    _14 = is_op("add")(_13, is_constant())
    _15 = is_op("power")(_14, is_constant())
    _16 = is_op("multiply")(_15, is_constant())
    _17 = is_op("multiply")(_10, _16)
    _18 = is_op("multiply")(_9, _16)
    _19 = is_op("subtract")(is_constant(), _17)
    _20 = is_op("add")(_18, _19)
    _21 = is_op("reshape")(_20)
    return _21

def attention():
    tmp = wildcard()
    _22 = is_op("nn.dense")(tmp, is_constant())
    _23 = is_op("add")(_22, is_constant())
    _24 = is_op("reshape")(_23)
    _25 = is_op("transpose")(_24)
    _26 = is_op("nn.dense")(tmp, is_constant())
    _27 = is_op("add")(_26, is_constant())
    _28 = is_op("reshape")(_27)
    _29 = is_op("transpose")(_28)
    _30 = is_op("reshape")(_25)
    _31 = is_op("reshape")(_29)
    _32 = is_op("nn.batch_matmul")(_30, _31)
    _33 = is_op("reshape")(_32)
    _37 = is_op("expand_dims")(wildcard())
    _38 = is_op("subtract")(is_constant(), _37)
    _39 = is_op("multiply")(_33, is_constant())
    _40 = is_op("multiply")(_38, is_constant())
    _41 = is_op("add")(_39, _40)
    _42 = is_op("nn.softmax")(_41)
    _43 = is_op("nn.dense")(tmp, is_constant())
    _44 = is_op("add")(_43, is_constant())
    _45 = is_op("reshape")(_44)
    _46 = is_op("transpose")(_45)
    _47 = is_op("reshape")(_46)
    _48 = is_op("reshape")(_42)
    _49 = is_op("transpose")(_47)
    _50 = is_op("nn.batch_matmul")(_48, _49)
    _51 = is_op("reshape")(_50)
    _52 = is_op("transpose")(_51)
    _53 = is_op("reshape")(_52)
    _54 = is_op("nn.dense")(_53, is_constant())
    _55 = is_op("add")(_54, is_constant())
    _56 = is_op("add")(_55, tmp)
    return _56

def mask():
    _34 = is_op("reshape")(wildcard())
    _35 = is_op("cast")(_34)
    _36 = is_op("multiply")(is_constant(), _35)
    return _36

def layernorm():
    tmp = wildcard()
    _57 = is_op("mean")(tmp)
    _58 = is_op("subtract")(tmp, _57)
    _59 = is_op("multiply")(_58, _58)
    _60 = is_op("mean")(_59)
    _61 = is_op("add")(_60, is_constant())
    _62 = is_op("power")(_61, is_constant())
    _63 = is_op("multiply")(_62, is_constant())
    _64 = is_op("multiply")(_57, _63)
    _65 = is_op("multiply")(tmp, _63)
    _66 = is_op("subtract")(is_constant(), _64)
    _67 = is_op("add")(_65, _66)
    return _67

def intermediate():
    tmp = wildcard()
    _68 = is_op("nn.dense")(tmp, is_constant())
    _69 = is_op("add")(_68, is_constant())
    _70 = is_op("power")(_69, is_constant())
    _71 = is_op("multiply")(is_constant(), _70)
    _72 = is_op("add")(_69, _71)
    _73 = is_op("multiply")(is_constant(), _72)
    _74 = is_op("tanh")(_73)
    _75 = is_op("add")(is_constant(), _74)
    _76 = is_op("multiply")(is_constant(), _75)
    _77 = is_op("multiply")(_69, _76)
    _78 = is_op("nn.dense")(_77, is_constant())
    _79 = is_op("add")(_78, is_constant())
    _80 = is_op("add")(_79, tmp)
    return _80

def end():
    return

class pipeline_tensor_partition(ExprMutator):
    def __init__(self, pipeline, tensor):
        self._pipeline = pipeline
        self._tensor = tensor
        super().__init__()
    
        

if __name__ == '__main__':
    # run_tf(model_path)
    layout = "NHWC"
    mod, params = create_graph(model_path)
    # print("-------------original model--------------")
    # print(mod["main"].astext(show_meta_data=False))
    
    mod, params = relay.optimize(mod, "llvm", params)
    mod = transform.DefuseOps()(mod)
    # print("-------------optimized model-------------")
    # print(mod["main"].astext(show_meta_data=False))
    embedding_p = embedding()
    attention_p = attention()
    mask_p = mask()
    layernorm_p = layernorm()
    intermediate_p = intermediate()
    
    partitioned = embedding_p.partition(mod["main"], {"Composite": "embedding"})
    partitioned = attention_p.partition(partitioned, {"Composite": "attention"})
    partitioned = mask_p.partition(partitioned, {"Composite": "mask"})
    partitioned = layernorm_p.partition(partitioned, {"Composite": "layernorm"})
    partitioned = intermediate_p.partition(partitioned, {"Composite": "intermediate"})

    mod["main"] = partitioned
    print(mod["main"].astext(show_meta_data=False))

    
    # target = "llvm"
    # dev = tvm.device(target, 0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target, params=params)

    # out_float = run_test(lib)
    # origin_shape = np.array(out_float.shape)
    # print('*'*100, origin_shape)

    # with tvm.transform.PassContext(opt_level=3):
    #    lib = relay.build(mod, target, params=params)

    # out_float = run_test(lib)
    # run_inference(lib)


    # target = "aipu" # "llvm"
    # dev = tvm.device(target, 0)

    # mod_quantized = quantize(mod, params, data_aware=True)
    # print("-------------mod_quantized model--------------")
    # print(mod_quantized["main"].astext(show_meta_data=False))
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target, params=params)


    # aipu_output = run_test(lib)
    # # run_inference(lib, origin_shape)

    # print("aipu error rate: ", np.sum(np.abs(out_float - aipu_output )) / np.sum(np.abs(out_float)))
    # print("bert inferrence done !!!")
    # print("-------------relay.build end--------------")
    # # out_int8 = run_test(lib)
    # # print(np.sum(np.abs(out_float - out_int8)) / np.sum(np.abs(out_float)))
    # # run_inference(lib)

    # savedata = out_float
    # print(out_float.shape, savedata.shape)
    # savedata = np.reshape(savedata, (-1, 1))
    # np.savetxt("./txt/llvm/bert.txt", savedata, fmt="%.10f")
    # print(savedata)
    # cpparr = np.loadtxt("./txt/tmp/bert_1.txt")
    # print(out_float.shape, savedata.shape, cpparr.shape)
    # cpparr = cpparr.reshape(savedata.shape)
    # print("cpp error rate: ", np.sum(np.abs((savedata) - (cpparr))) / np.sum(np.abs(savedata)))
    # print("yolov3 inferrence done !!!")

