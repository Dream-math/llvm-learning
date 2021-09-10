# Given a list which describes operators executing environment (run on some device),
# then we can use relay.transform.MergeCompilerRegions to annotate the whole model,
# then do the partition by transform.PartitionGraph

import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import run_opt_pass
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.expr import Call

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
    cb_1 = compiler_begin(data, "test")
    O_1 = relay.abs(cb_1)
    ce_1 = compiler_end(O_1, "test")
    ce_2 = compiler_end(O_1, "test")
    cb_2 = compiler_begin(ce_1, "test")
    cb_3 = compiler_begin(ce_2, "default")
    O_2 = relay.nn.relu(cb_2)
    ce_3 = compiler_end(O_2, "test")

    X = relay.tanh(cb_3)
    ce_4 = compiler_end(X, "default")

    cb_4 = compiler_begin(ce_3, "test")
    cb_5 = compiler_begin(ce_4, "test")
    O_3 = relay.add(cb_4, cb_5)
    ce_5 = compiler_end(O_3, "test")

    diamond = relay.Function([data], ce_5)
    return diamond

def expected():
    data = relay.var("data", shape=(10, 10))
    cb_1 = compiler_begin(data, "test")
    O_1 = relay.abs(cb_1)
    ce_2 = compiler_end(O_1, "test")
    O_2 = relay.nn.relu(O_1)
    ce_3 = compiler_end(O_2, "test")

    cb_3 = compiler_begin(ce_2, "default")
    X = relay.tanh(cb_3)
    ce_4 = compiler_end(X, "default")

    cb_4 = compiler_begin(ce_3, "test")
    cb_5 = compiler_begin(ce_4, "test")
    O_3 = relay.add(cb_4, cb_5)
    ce_5 = compiler_end(O_3, "test")

    func = relay.Function([data], ce_5)
    return func

class ScheduleConv2d(ExprMutator):
    def __init__(self, compiler):
        self.compiler = compiler
        super().__init__()

    def visit_call(self, expr):
        new_args = []
        new_fn = self.visit(expr.op)
        if expr.op != tvm.relay.op.get("tanh"):
            for arg in expr.args:
                if isinstance(arg, Call):
                    if arg.op != tvm.relay.op.get("tanh"):
                        arg = compiler_end(self.visit(arg), self.compiler)
                    else:
                        arg = compiler_end(self.visit(arg), "default")
                arg = compiler_begin(arg, self.compiler)
                new_args.append(arg)
        else:
            for arg in expr.args:
                if isinstance(arg, Call):
                    if arg.op != tvm.relay.op.get("tanh"):
                        arg = compiler_end(self.visit(arg), self.compiler)
                    else:
                        arg = compiler_end(self.visit(arg), "default")
                arg = compiler_begin(arg, "default")
                new_args.append(arg)
        return Call(new_fn, new_args, expr.attrs, expr.type_args, expr.span)

def annotate(sched, fn):
    new_fn = sched.visit(fn)
    new_params = [x for x in new_fn.params]
    # add compiler_end for the last operator in fn
    if (new_fn.body.op != tvm.relay.op.get("tanh")):
        new_body = compiler_end(new_fn.body, sched.compiler)
    else:
        new_body = compiler_end(new_fn.body, "default")
    return relay.Function(list(new_params), new_body, new_fn.ret_type, new_fn.type_params, new_fn.attrs)


func = origin()
sched = ScheduleConv2d("test")
func = annotate(sched, func)
diamond = diamond_graph_fanouts()
# print(func)
# print(diamond)
assert tvm.ir.structural_equal(func, diamond)
func = run_opt_pass(func, relay.transform.MergeCompilerRegions())
mod = tvm.IRModule.from_expr(func)
mod = transform.PartitionGraph()(mod)
print(mod.astext(show_meta_data=False))
