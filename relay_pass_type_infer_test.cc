/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <gtest/gtest.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/te/operation.h>
#include <tvm/relay/attrs/annotation.h>

std::set<std::string> operator_set = {"nn.relu", "add"};
using Expr = tvm::RelayExpr;
using CallNode = tvm::relay::CallNode;
using OpNode = tvm::OpNode;
using Op = tvm::Op;

class CustomizedAnnotation : tvm::relay::ExprMutator {
public:
  explicit CustomizedAnnotation(std::string compiler)
    : compiler_(compiler) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    auto compiler_begin = tvm::relay::Op::Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object
      <tvm::relay::CompilerAttrs>();
    attrs_compiler->compiler = compiler_;
    auto attrs_default = tvm::runtime::make_object
      <tvm::relay::CompilerAttrs>();
    attrs_default->compiler = "default";
    tvm::Array<Expr> call_args;
    auto new_op = VisitExpr(call_node->op);
    for (auto arg : call_node->args) {
      if (arg->IsInstance<CallNode>()){
        const CallNode * node = (static_cast<const CallNode *>
                                 (arg.get()));
        const auto op_node = node->op.as<OpNode>();
        std::string argop_name = tvm::runtime::GetRef<Op>(op_node)->name;
        if (operator_set.find(argop_name) != operator_set.end())
          arg = tvm::relay::Call(compiler_end, {VisitExpr(arg)},
                                 tvm::Attrs(attrs_compiler), {});
        else
          arg = tvm::relay::Call(compiler_end, {VisitExpr(arg)},
                                 tvm::Attrs(attrs_default), {});
      }
      const auto callop_node = call_node->op.as<OpNode>();
      std::string callop_name = tvm::runtime::GetRef<Op>(callop_node)->
        name;
      if (operator_set.find(callop_name) != operator_set.end())
        arg = tvm::relay::Call(compiler_begin, {arg},
                                   tvm::Attrs(attrs_compiler), {});
      else
        arg = tvm::relay::Call(compiler_begin, {arg},
                                   tvm::Attrs(attrs_default), {});
      call_args.push_back(arg);
    }
    return tvm::relay::Call(new_op, call_args, call_node->attrs,
                       call_node->type_args, call_node->span);
  }

  tvm::relay::Function annotate(tvm::relay::Function func){
    auto compiler_begin = tvm::relay::Op::
      Get("annotation.compiler_begin");
    auto compiler_end = tvm::relay::Op::Get("annotation.compiler_end");
    auto attrs_compiler = tvm::runtime::make_object
      <tvm::relay::CompilerAttrs>();
    attrs_compiler->compiler = compiler_;
    auto attrs_default = tvm::runtime::make_object
      <tvm::relay::CompilerAttrs>();
    attrs_default->compiler = "default";
    auto new_expr = VisitExpr(func);
    const tvm::relay::FunctionNode * new_funcnode= static_cast
      <const tvm::relay::FunctionNode *>(new_expr.get());
    const CallNode * node = (static_cast<const CallNode *>
                             (new_funcnode->body.get()));
    const auto op_node = node->op.as<OpNode>();
    std::string name = tvm::runtime::GetRef<Op>(op_node)->name;
    auto new_body = tvm::relay::Call();
    if (operator_set.find(name) != operator_set.end())
      {
        new_body = tvm::relay::Call(compiler_end, {new_funcnode->body},
                                    tvm::Attrs(attrs_compiler), {});
      }
    else
      {
        new_body = tvm::relay::Call(compiler_end, {new_funcnode->body},
                                    tvm::Attrs(attrs_default), {});
      }
    auto new_func = tvm::runtime::GetRef<tvm::relay::Function>(new_funcnode);
    return tvm::relay::Function(tvm::relay::FreeVars(new_body), new_body,
                           new_func->ret_type, new_func->type_params,
                           new_func->attrs);
  }

  std::string compiler_;
};


TEST(Relay, SelfReference) {
  using namespace tvm;
  auto tensor_type = relay::TensorType({1, 1, 10, 10},
                                       DataType::Float(32));
  auto data = relay::Var("data", tensor_type);
  auto abs_op = relay::Op::Get("abs");
  auto add_op = relay::Op::Get("add");
  auto relu_op = relay::Op::Get("nn.relu");
  auto tanh_op = relay::Op::Get("tanh");
  // auto compiler_begin = relay::Op::Get("annotation.compiler_begin");
  // auto compiler_end = relay::Op::Get("annotation.compiler_end");
  // auto attrs_dnnl = make_object<tvm::relay::CompilerAttrs>();
  // attrs_dnnl->compiler = "dnnl";
  // auto attrs_default = make_object<tvm::relay::CompilerAttrs>();
  // attrs_default->compiler = "default";
  // auto O1 = relay::Call(compiler_begin, {data}, Attrs(attrs_default), {});
  // auto O2 = relay::Call(abs_op, {O1}, tvm::Attrs(), {});
  // auto O3 = relay::Call(compiler_end, {O2}, Attrs(attrs_default), {});
  // auto O4 = relay::Call(compiler_end, {O2}, Attrs(attrs_default), {});
  // auto O5 = relay::Call(compiler_begin, {O3}, Attrs(attrs_dnnl), {});
  // auto O6 = relay::Call(compiler_begin, {O4}, Attrs(attrs_default), {});
  // auto O7 = relay::Call(relu_op, {O5}, tvm::Attrs(), {});
  // auto O8 = relay::Call(compiler_end, {O7}, Attrs(attrs_dnnl), {});
  // auto O9 = relay::Call(tanh_op, {O6}, tvm::Attrs(), {});
  // auto O10 = relay::Call(compiler_end, {O9}, Attrs(attrs_default), {});
  // auto O11 = relay::Call(compiler_begin, {O8}, Attrs(attrs_dnnl), {});
  // auto O12 = relay::Call(compiler_begin, {O10}, Attrs(attrs_dnnl), {});
  // auto O13 = relay::Call(add_op, {O11, O12}, tvm::Attrs(), {});
  // auto O14 = relay::Call(compiler_end, {O13}, Attrs(attrs_dnnl), {});
  auto O_1 = relay::Call(abs_op, {data}, tvm::Attrs(), {});
  auto O_2 = relay::Call(relu_op, {O_1}, tvm::Attrs(), {});
  auto O_3 = relay::Call(tanh_op, {O_1}, tvm::Attrs(), {});
  auto O_4 = relay::Call(add_op, {O_2, O_3}, tvm::Attrs(), {});
  auto func = relay::Function(relay::FreeVars(O_4), O_4, relay::Type(),
                              {});
  CustomizedAnnotation custom("dnnl");
  auto new_func = custom.annotate(func);
  LOG(INFO) << AsText(new_func, false);
  auto mod = IRModule::FromExpr(new_func);
  // auto mod = IRModule::FromExpr(func);
  mod = relay::transform::InferType()(mod);
  // auto merge_pass = relay::transform::MergeCompilerRegions();
  mod = relay::transform::MergeCompilerRegions()(mod);
  mod = relay::transform::PartitionGraph()(mod);
  // auto fx = mod->Lookup("main");
  LOG(INFO) << AsText(mod, false);

  // ICHECK(tvm::StructuralEqual()(type_fx->checked_type(), expected));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
