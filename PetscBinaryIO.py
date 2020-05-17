# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:53:16 2020

@author: Darin
"""

from struct import pack, unpack
import numpy as np
import scipy.sparse as sparse

def PetscBinaryRead(file, indSize=4, floatSize=8):
    intSize = 4
    floatChar = 'd'
    if floatSize == 4:
        floatChar = 'f'
    elif floatSize != 8:
        raise ValueError("Can only read 4-byte and 8-byte floats")
        
    with open(file, mode='rb') as fh:
        header = unpack('>i', fh.read(indSize))[0] # Read one int32 in big-endian format
        if header == 1211216: # Mat object
            m = unpack('>i', fh.read(indSize))[0]
            n = unpack('>i', fh.read(indSize))[0]
            nz = unpack('>i', fh.read(indSize))[0]
            if nz == -1: # Dense format
                s = np.array(unpack('>' + m*n*floatChar, fh.read(m*n*floatSize)))
                return s.reshape(m, n)
            else: # Sparse format
                nnz = np.array(unpack('>' + m*'i', fh.read(m*indSize)))
                sum_nz = nnz.sum()
                if sum_nz != nz:
                    raise ValueError("Number of nonzeros, %i, and rowlengths, "
                                     "%i, do not match" % (nz, sum_nz))
                j = np.array(unpack('>' + nz*'i', fh.read(nz*indSize)))
                s = np.array(unpack('>' + nz*floatChar, fh.read(nz*floatSize)))
                ind = np.concatenate([[0], nnz.cumsum()])
                return sparse.csr_matrix((s, j, ind), shape=(m, n))
            
            
        elif header == 1211214: # Vec object
            size = unpack('>i', fh.read(indSize))[0]
            return np.array(unpack('>' + size*floatChar, fh.read(size*floatSize)))
        
        elif header == 1211218: # IS object
            size = unpack('>i', fh.read(indSize))[0]
            return np.array(unpack('>' + size*'i'), fh.read(size*intSize))
        
        elif header == 1211221: # DM object
            m = np.array(unpack('>' + 12*'i', fh.read(12*intSize)))
            return 'DM %i by %i by %i' % (m[2], m[3], m[4])
        
        else:
            raise ValueError("Unrecognized header type")


def PetscBinaryWrite(file, v, intSize=4, floatSize=8):
        
    with open(file, mode='wb') as fh:
        if np.prod(v.shape) == v.size: # One non-unity dimension -> save as Vec
            fh.write(pack('>ii', 1211214, v.size))
            fh.write(pack('>' + v.size*'d', *v))
            
        else: # Save as Mat
            if hasattr(v, 'nnz'): # Sparse matrix
                fh.write(pack('>iiii', 1211216, v.shape[0], v.shape[1], v.nnz))
                v = v.tocsr()
                fh.write(pack('>' + v.shape[0]*'i', *(v.indptr[1:]-v.indptr[:-1])))
                fh.write(pack('>' + v.nnz*'i', *v.indices))
                fh.write(pack('>' + v.nnz*'d', *v.data))
            else: # Dense matrix
                fh.write(pack('>iiii', 1211216, v.shape[0], v.shape[1], v.size))
                fh.write(pack('>' + v.size*'d', *v.ravel()))
            
if __name__ == '__main__':
    vec = PetscBinaryRead('C:\\Users\\Darin\\Documents\\Blue Waters Results\\Stabiliy_Comparison_11211023.bw_GMG\\K')