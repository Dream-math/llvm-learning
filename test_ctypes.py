import ctypes
import pickle
from mpi4py import MPI
import numpy as np
import sys

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

a = (ctypes.c_int * 2)(1, 2)
b = ctypes.cast(a, ctypes.c_void_p)
# c = ctypes.cast(a, ctypes.POINTER(ctypes.c_char * 16))
mytype = MyType(b, 2)
handle = MyTypeHandle(mytype)
test = Test(handle, True)

sub_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['/home/hermanhe/tmp/trash/t2.py'], maxprocs=2)
sub_comm.bcast(test, MPI.ROOT)
sub_comm.Disconnect()
# with open('mydata', 'wb') as handle:
#     pickle.dump(test, handle)

# with open('mydata', 'rb') as handle:
#     y = pickle.load(handle)

# z = y.handle.contents.data
# z = (ctypes.c_int * 4).from_buffer_copy(y)
# z = ctypes.cast(z, ctypes.POINTER(ctypes.c_int))
# for i in range(2):
#     print(z[i])
    # print(test.handle.contents.data[i])

