# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:30:42 2019

@author: Darin
"""

import numpy as np


def QuadRule(nodes, ndims):
    """ Returns the Gauss quadrature points and weights for the specified element type
    
    Parameters
    ----------
    nodes : integer
        Number of nodes in the element
    ndims : integer
        Dimensionality of the elemnt (2 or 3)
    
    Returns
    -------
    GP : array_like
        Gauss quadrature points
    w : array_like
        Weights for each quadrature point
    
    """
    
    if ndims == 2:
        if nodes == 4:
            GP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
            w = np.ones(4)
    elif ndims == 3:
        if nodes == 8:
            GP = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) / np.sqrt(3)
            w = np.ones(8)
    
    return GP, w

def dN(quadPoint, nodes):
    """ Returns the shape function derivative at the provided quadrature points
    with respect to parent coordinates
    
    Parameters
    ----------
    quadPoint : array_like
        Parent coordinates where the shape function values are calculated.
        Size should be np x nd where np is number of quadrature points to
        evaluate and nd is dimensionality of the problem (2 or 3).
    nodes : integer
        Number of nodes in the element
    
    Returns
    -------
    dNdxi : array_like
        Values of the shape function derivatives
    
    """
    
    if len(quadPoint.shape) > 2:
        raise ValueError('quadPoint array needs to be size np x nd')
    elif len(quadPoint.shape) == 1:
        quadPoint = quadPoint.reshape(1,-1)
        
    dNdxi = np.zeros((quadPoint.shape[0], nodes, quadPoint.shape[1]))
    if quadPoint.shape[1] == 2:
        if nodes == 4:
            dNdxi[:, 0, 0] =  quadPoint[:,1] - 1
            dNdxi[:, 0, 1] =  quadPoint[:,0] - 1
            dNdxi[:, 1, 0] =  1 - quadPoint[:,1]
            dNdxi[:, 1, 1] = -1 - quadPoint[:,0]
            dNdxi[:, 2, 0] =  1 + quadPoint[:,1]
            dNdxi[:, 2, 1] =  1 + quadPoint[:,0]
            dNdxi[:, 3, 0] = -1 - quadPoint[:,1]
            dNdxi[:, 3, 1] =  1 - quadPoint[:,0]
            dNdxi /= 4
        else:
            raise ValueError("Shape functions not implemented for %i"%nodes +
                             " nodes in 2D")
    if quadPoint.shape[1] == 3:
        if nodes == 8:
            # N1 = 1/8 * (1-xi) * (1-eta) * (1-zeta)
            dNdxi[:, 0, 0] = -(1 - quadPoint[:,1]) * (1 - quadPoint[:,2])
            dNdxi[:, 0, 1] = -(1 - quadPoint[:,0]) * (1 - quadPoint[:,2])
            dNdxi[:, 0, 2] = -(1 - quadPoint[:,0]) * (1 - quadPoint[:,1])
            # N2 = 1/8 * (1+xi) * (1-eta) * (1-zeta)
            dNdxi[:, 1, 0] =  (1 - quadPoint[:,1]) * (1 - quadPoint[:,2])
            dNdxi[:, 1, 1] = -(1 + quadPoint[:,0]) * (1 - quadPoint[:,2])
            dNdxi[:, 1, 2] = -(1 + quadPoint[:,0]) * (1 - quadPoint[:,1])
            # N3 = 1/8 * (1+xi) * (1+eta) * (1-zeta)
            dNdxi[:, 2, 0] =  (1 + quadPoint[:,1]) * (1 - quadPoint[:,2])
            dNdxi[:, 2, 1] =  (1 + quadPoint[:,0]) * (1 - quadPoint[:,2])
            dNdxi[:, 2, 2] = -(1 + quadPoint[:,0]) * (1 + quadPoint[:,1])
            # N4 = 1/8 * (1-xi) * (1+eta) * (1-zeta)
            dNdxi[:, 3, 0] = -(1 + quadPoint[:,1]) * (1 - quadPoint[:,2])
            dNdxi[:, 3, 1] =  (1 - quadPoint[:,0]) * (1 - quadPoint[:,2])
            dNdxi[:, 3, 2] = -(1 - quadPoint[:,0]) * (1 + quadPoint[:,1])
            # N5 = 1/8 * (1-xi) * (1-eta) * (1+zeta)
            dNdxi[:, 4, 0] = -(1 - quadPoint[:,1]) * (1 + quadPoint[:,2])
            dNdxi[:, 4, 1] = -(1 - quadPoint[:,0]) * (1 + quadPoint[:,2])
            dNdxi[:, 4, 2] =  (1 - quadPoint[:,0]) * (1 - quadPoint[:,1])
            # N6 = 1/8 * (1+xi) * (1-eta) * (1+zeta)
            dNdxi[:, 5, 0] =  (1 - quadPoint[:,1]) * (1 + quadPoint[:,2])
            dNdxi[:, 5, 1] = -(1 + quadPoint[:,0]) * (1 + quadPoint[:,2])
            dNdxi[:, 5, 2] =  (1 + quadPoint[:,0]) * (1 - quadPoint[:,1])
            # N7 = 1/8 * (1+xi) * (1+eta) * (1+zeta)
            dNdxi[:, 6, 0] =  (1 + quadPoint[:,1]) * (1 + quadPoint[:,2])
            dNdxi[:, 6, 1] =  (1 + quadPoint[:,0]) * (1 + quadPoint[:,2])
            dNdxi[:, 6, 2] =  (1 + quadPoint[:,0]) * (1 + quadPoint[:,1])
            # N8 = 1/8 * (1-xi) * (1+eta) * (1+zeta)
            dNdxi[:, 7, 0] = -(1 + quadPoint[:,1]) * (1 + quadPoint[:,2])
            dNdxi[:, 7, 1] =  (1 - quadPoint[:,0]) * (1 + quadPoint[:,2])
            dNdxi[:, 7, 2] =  (1 - quadPoint[:,0]) * (1 + quadPoint[:,1])
            dNdxi /= 8
        else:
            raise ValueError("Shape functions not implemented for %i"%nodes +
                             " nodes in 3D")
            
    return dNdxi

def AssembleB(dNdx):
    """ Assembles the B-matrix of shape function derivatives
    
    Parameters
    ----------
    dNdx : array_like
        Shape function derivatives
    
    Returns
    -------
    B : array_like
        B-matrix of shape function derivatives
    
    """
    
    if dNdx.shape[0] == 2:
        B = np.zeros((3, 2*dNdx.shape[1]))
        B[0, ::2] =  dNdx[0, :]
        B[1, 1::2] = dNdx[1, :]
        B[2, ::2] =  dNdx[1, :]
        B[2, 1::2] = dNdx[0, :]
    elif dNdx.shape[0] == 3:
        B = np.zeros((6, 3*dNdx.shape[1]))
        B[0, ::3] =  dNdx[0, :]
        B[1, 1::3] = dNdx[1, :]
        B[2, 2::3] = dNdx[2, :]
        B[3, ::3] =  dNdx[1, :]
        B[3, 1::3] = dNdx[0, :]
        B[4, 1::3] = dNdx[2, :]
        B[4, 2::3] = dNdx[1, :]
        B[5, ::3] =  dNdx[2, :]
        B[5, 2::3] = dNdx[0, :]
    else:
        raise ValueError("dNdx has unexpected shape")
        
    return B

def AssembleG(dNdx):
    """ Assembles the G-matrix of shape function derivatives used in stress
    stiffness matrix assembly
    
    Parameters
    ----------
    dNdx : array_like
        Shape function derivatives
    
    Returns
    -------
    G : array_like
        G-matrix of shape function derivatives
    
    """
    
    G = np.zeros((dNdx.shape[0], dNdx.size))
    G[:dNdx.shape[0],::dNdx.shape[0]] = dNdx
    
    # This is the "right" way to do it, but we'll save time by omitting some entries
#    G = np.zeros((dNdx.shape[0]**2, dNdx.size))
#    for i in range(dNdx.shape[0]):
#        G[i*dNdx.shape[0]:(i+1)*dNdx.shape[0],i::dNdx.shape[0]] = dNdx
        
    return G