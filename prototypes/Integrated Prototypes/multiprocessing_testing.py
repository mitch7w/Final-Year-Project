import multiprocessing
import ctypes
import numpy as np
import time

def foo(q,classifier_result):
    while True:
        if not q.empty():
            classifier_result.value = q.get()

def main_loop(q,classifier_result):
    while True:
        q.put(np.random.rand())
     
if __name__ == "__main__":
    q = multiprocessing.Queue()
    classifier_result = multiprocessing.Value(ctypes.c_float,2)
    p1 = multiprocessing.Process(target=foo, args=(q,classifier_result))
    p2 = multiprocessing.Process(target=main_loop, args=(q,classifier_result))

    p1.start()
    p2.start()

    while True:
        print("classifier_result: ", classifier_result.value)
    # while True:
    #     print("classifier_result: ", classifier_result.value)