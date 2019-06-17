import random
import numpy as np

class MemoryReplay(object):

    def __init__(self,
                 max_size=10000,
                 bs=64,
                 im_size=84,
                 stack=4,
                 num_heads = 10,
                 p = 0.5):

        self.s = np.zeros((max_size, stack+1, im_size, im_size), dtype=np.float32)
        self.r = np.zeros(max_size, dtype=np.float32)
        self.a = np.zeros(max_size, dtype=np.int32)
        #self.ss = np.zeros_like(self.s)
        self.done = np.array([True]*max_size)
        self.head_mask = np.zeros((max_size, num_heads), dtype=np.int32)
        
        self.num_heads = num_heads
        self.p = p
        self.stack = stack
        self.max_size = max_size
        self.bs = bs
        self._cursor = None
#        self.total_idx = list(range(self.max_size))


    def put(self, sras):

        if self._cursor == (self.max_size-1) or self._cursor is None :
            self._cursor = 0
        else:
            self._cursor += 1

        self.s[self._cursor] = sras[0]
        self.a[self._cursor] = sras[1]
        self.r[self._cursor] = sras[2]
        #self.ss[self._cursor] = sras[3]
        self.done[self._cursor] = sras[3]
        self.head_mask[self._cursor] = np.random.binomial(1, self.p, self.num_heads)


    def batch(self,k):
        idx =  np.array(range(self.max_size))
        idx_k = self.head_mask[:,k].astype(bool)
        sample_idx = random.sample(list(idx[idx_k]), self.bs)
        s = self.s[sample_idx, : self.stack]
        a = self.a[sample_idx]
        r = self.r[sample_idx]
        #ss = self.ss[sample_idx]
        ss = self.s[sample_idx, 1:]
        done = self.done[sample_idx]
#        head_mask = self.head_mask[sample_idx]

        return s, a, r, ss, done
