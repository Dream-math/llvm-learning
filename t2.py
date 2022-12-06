from mpi4py import MPI
import numpy as np
import tvm

import ctypes
import pickle

MyTypeData = ctypes.POINTER(ctypes.c_int)
class MyType(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", ctypes.c_int),
    ]

MyTypeHandle = ctypes.POINTER(MyType)

class Test:
    def __init__(self, handle, flag):
        self.handle = handle
        self.flag = flag

    def __getstate__(self):
        tmp_dict = {}
        # tmp_dict["data"] = bytes((ctypes.c_ubyte * 8).from_address(ctypes.c_void_p.from_buffer(self.handle.contents.data).value))
        tmp_dict["data"] = bytes((ctypes.c_ubyte * 8).from_address(self.handle.contents.data))
        tmp_dict["device"] = self.handle.contents.device
        tmp_dict["flag"] = self.flag
        return tmp_dict

    def __setstate__(self, arg):
        self.flag = arg["flag"]
        cvoidp = (ctypes.c_ubyte * 8).from_buffer_copy(arg["data"])
        # cvoidp = MyTypeData.from_buffer_copy(arg["data"])
        cvoidp = ctypes.cast(cvoidp, ctypes.c_void_p)
        tmpmytype = MyType(cvoidp, arg["device"])
        self.handle = MyTypeHandle(tmpmytype)

comm = MPI.Comm.Get_parent()
# print(comm.Get_rank())
# print(comm.Get_size())

# print 'ho ', comm.Get_rank(), ' of  ',comm.Get_size(),' ', MPI.COMM_WORLD.Get_rank(), ' of  ',MPI.COMM_WORLD.Get_size()
# print(MPI.COMM_WORLD.Get_rank())
# a = MPI.COMM_WORLD.bcast(MPI.COMM_WORLD.Get_rank(), root=1)
# print("value from other child ", a)

# if comm.Get_rank() == 0:
# with open('mydata', 'rb') as handle:
#     y = pickle.load(handle)
# print(y.handle.contents.device)
# z = ctypes.cast(ctypes.c_void_p(y.handle.contents.data), ctypes.POINTER(ctypes.c_int))
# for i in range(4):
#     print(z[i])

data = comm.bcast(None, root=0)
print(type(data))
# print("value from parent", data)

# z = ctypes.cast(data.handle.contents.data, MyTypeData)
# print(data.flag)
# z = ctypes.cast(ctypes.c_void_p(y.handle.contents.data), ctypes.POINTER(ctypes.c_int))
# for i in range(2):
#     print(z[i])
# for i in range(4):
#     print(z[i])
# data = data * comm.Get_rank()
# print(data)

# common_comm=comm.Merge(True)
# count = [4, 4]
# displ = [0, 4]
# comm.Gatherv(data.flatten(), [None, count, displ, MPI.FLOAT], root=0)
# print("common_comm.Is_inter", common_comm.Is_inter())
# print 'common_comm ', common_comm.Get_rank(), ' of  ',common_comm.Get_size()
# print(common_comm.Get_rank())
# c=common_comm.bcast(1, root=0)
# print("value from rank 0 in common_comm", c)
comm.Disconnect()
