# Given a list which describes operators executing environment (run on some device),
# then we can use relay.transform.MergeCompilerRegions to annotate the whole model,
# then do the partition by transform.PartitionGraph

# since we rely on relay.transform.MergeCompilerRegions, it can not guarantee the
# partition is optimal

import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import run_opt_pass
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.expr import Call

# operators execute on dnnl
operator_list = ["nn.relu", "add"]

def origin():
    data = relay.var("data", shape=(10, 10))
    O_1 = relay.abs(data)
    O_2 = relay.nn.relu(O_1)
    X = relay.tanh(O_1)
    O_3 = relay.add(O_2, X)
    diamond = relay.Function([data], O_3)
    return diamond

def diamond_graph_fanouts():
    data = relay.var("data", shape=(10, 10))
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


func = origin()
anno = CustomizedAnnotation("dnnl")
func = annotate(anno, func)
diamond = diamond_graph_fanouts()
assert tvm.ir.structural_equal(func, diamond)
func = run_opt_pass(func, relay.transform.MergeCompilerRegions())
mod = tvm.IRModule.from_expr(func)
mod = transform.PartitionGraph()(mod)
print(mod.astext(show_meta_data=True))
