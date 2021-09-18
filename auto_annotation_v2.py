# Given a list which describes operators executing environment (run on some device),
# then we can use relay.transform.MergeCompilerRegions to annotate the whole model,
# then do the partition by transform.PartitionGraph

# since we rely on relay.transform.MergeCompilerRegions, it can not guarantee the
# partition is optimal

import os
import sys
import numpy as np

import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import run_opt_pass
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.expr import Call
from tvm.contrib import utils
from tvm import runtime

# operators execute on dnnl
operator_list = ["nn.relu", "add"]


def check_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu(), params=None
):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        source_dir = os.getenv('TVM_HOME')
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
        tmp_path = utils.tempdir()
        lib_name = "lib.so"
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe, device)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_executor_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            json, lib, param = relay.build(mod, target=target, params=params)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, device=device)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)
    check_vm_result()
    check_graph_executor_result()

def origin(shape, dtype):
    data = relay.var("data", shape=shape, dtype=dtype)
    O_1 = relay.abs(data)
    O_2 = relay.nn.relu(O_1)
    X = relay.tanh(O_1)
    O_3 = relay.add(O_2, X)
    diamond = relay.Function([data], O_3)
    return diamond

def diamond_graph_fanouts(shape, dtype):
    data = relay.var("data", shape=shape, dtype=dtype)
    cb_1 = compiler_begin(data, "default")
    O_1 = relay.abs(cb_1)
    ce_1 = compiler_end(O_1, "default")
    ce_2 = compiler_end(O_1, "default")
    cb_2 = compiler_begin(ce_1, "dnnl")
    cb_3 = compiler_begin(ce_2, "default")
    O_2 = relay.nn.relu(cb_2)
    ce_3 = compiler_end(O_2, "dnnl")

    X = relay.tanh(cb_3)
    ce_4 = compiler_end(X, "default")

    cb_4 = compiler_begin(ce_3, "dnnl")
    cb_5 = compiler_begin(ce_4, "dnnl")
    O_3 = relay.add(cb_4, cb_5)
    ce_5 = compiler_end(O_3, "dnnl")

    diamond = relay.Function([data], ce_5)
    return diamond

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
                    arg = compiler_end(self.visit(arg), "default")
            if expr.op.name in operator_list:
                arg = compiler_begin(arg, self.compiler)
            else:
                arg = compiler_begin(arg, "default")
            new_args.append(arg)
        return Call(new_fn, new_args, expr.attrs, expr.type_args, expr.span)


def annotate(anno, fn):
    new_fn = anno.visit(fn)
    new_params = [x for x in new_fn.params]
    # add compiler_end for the last operator in fn
    if new_fn.body.op.name in operator_list:
        new_body = compiler_end(new_fn.body, anno.compiler)
    else:
        new_body = compiler_end(new_fn.body, "default")
    return relay.Function(list(new_params), new_body, new_fn.ret_type, new_fn.type_params, new_fn.attrs)

def test_extern_dnnl():
    def annotated(dtype, ishape, w1shape):
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(
            data, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        depthwise_conv2d_2 = relay.nn.conv2d(
            depthwise_conv2d_1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        f = relay.Function([data, weight1], out)

        mod = tvm.IRModule.from_expr(f)
        return mod

    def expected(dtype, ishape, w1shape):
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        begin0 = relay.annotation.compiler_begin(data, "dnnl")
        begin1 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_1 = relay.nn.conv2d(
            begin0, begin1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        end0 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        end1 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        begin2 = relay.annotation.compiler_begin(end1, "dnnl")
        begin3 = relay.annotation.compiler_begin(end0, "dnnl")
        begin4 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_2 = relay.nn.conv2d(
            begin3, begin4, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        end2 = relay.annotation.compiler_end(depthwise_conv2d_2, "dnnl")
        begin5 = relay.annotation.compiler_begin(end2, "dnnl")
        out = relay.add(begin2, begin5)
        end3 = relay.annotation.compiler_end(out, "dnnl")
        f = relay.Function([data, weight1], end3)
        mod = tvm.IRModule.from_expr(f)
        return mod

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)

    def test_annotate():
        mod = annotated(dtype, ishape, w1shape)
        mod = transform.AnnotateTarget("dnnl")(mod)
        mod = relay.transform.InferType()(mod)
        ref_mod = expected(dtype, ishape, w1shape)
        ref_mod = relay.transform.InferType()(ref_mod)
        tvm.ir.assert_structural_equal(mod, ref_mod)

    def test_run():
        if not tvm.get_global_func("relay.ext.dnnl", True):
            print("skip because DNNL codegen is not available")
            return

        ref_mod = annotated(dtype, ishape, w1shape)
        mod = annotated(dtype, ishape, w1shape)
        mod = transform.AnnotateTarget("dnnl")(mod)
        mod = transform.PartitionGraph()(mod)
        # print(mod.astext(show_meta_data=False))

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        ref_ex = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu())
        ref_res = ref_ex.evaluate()(i_data, w1_data)

        check_result(
            mod, {"data": i_data, "weight1": w1_data}, (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5
        )

    test_annotate()
    test_run()

def test_auto_annotation():
    shape = (1, 1, 10, 10)
    dtype = "float32"

    func = origin(shape, dtype)
    anno = CustomizedAnnotation("dnnl")
    func = annotate(anno, func)
    diamond = diamond_graph_fanouts(shape, dtype)
    assert tvm.ir.structural_equal(func, diamond)
    func = run_opt_pass(func, relay.transform.MergeCompilerRegions())
    # diamond = run_opt_pass(diamond, relay.transform.MergeCompilerRegions())
    mod = tvm.IRModule.from_expr(func)
    # mod = tvm.IRModule.from_expr(diamond)
    mod = relay.transform.InferType()(mod)
    mod = transform.PartitionGraph()(mod)

    ref_func = origin(shape, dtype)
    ref_mod = tvm.IRModule.from_expr(ref_func)

    data = np.random.uniform(0, 1, shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu())
    ref_res = ref_ex.evaluate()(data)

    check_result(
        mod, {"data": data}, shape, ref_res.asnumpy(), tol=1e-5
    )

test_auto_annotation()
# test_extern_dnnl()
