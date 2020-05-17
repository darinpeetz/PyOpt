# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 14:20:55 2018

@author: Darin
"""

import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import inv, eigh, cholesky
from scipy.sparse.linalg import aslinearoperator, LinearOperator
import sys

# For intermediate plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# For plotting
class UpdatablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths
    
def _makeOperator(operatorInput, expectedShape):
    """Takes a dense numpy array or a sparse matrix or
    a function and makes an operator performing matrix * blockvector
    products.

    Examples
    --------
    >>> A = _makeOperator( arrayA, (n, n) )
    >>> vectorB = A( vectorX )

    """
    if operatorInput is None:
        def ident(x):
            return x
        operator = LinearOperator(expectedShape, ident, matmat=ident)
    else:
        operator = aslinearoperator(operatorInput)

    if operator.shape != expectedShape:
        raise ValueError('operator has invalid shape')

    return operator

def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False):
    """Orthonormalize with respect to B matrix"""
    if blockVectorBV is None:
        if B is not None:
            blockVectorBV = B(blockVectorV)
        else:
            blockVectorBV = blockVectorV  # Shared data!!!
    gramVBV = np.dot(blockVectorV.T, blockVectorBV)
    gramVBV = cholesky(gramVBV)
    gramVBV = inv(gramVBV, overwrite_a=True)
    # gramVBV is now R^{-1}.
    blockVectorV = np.dot(blockVectorV, gramVBV)
    if B is not None:
        blockVectorBV = np.dot(blockVectorBV, gramVBV)

    if retInvR:
        return blockVectorV, blockVectorBV, gramVBV
    else:
        return blockVectorV, blockVectorBV
    
def _sorteig(S, W, which, sigma):
    """Sort eigenvalues by proximity to target"""
    Scopy = S.copy()
    if sigma is not None:
        Scopy -= sigma
    if (which == "LM"):
        ind = np.argsort(np.abs(Scopy))
        S = S[ind[::-1]]
        W = W[:,ind[::-1]]
    elif (which == "SM"):
        ind = np.argsort(np.abs(Scopy))
        S = S[ind]
        W = W[:,ind]
    elif (which == "LR") or (which == "LA"):
        ind = np.argsort(Scopy)
        S = S[ind[::-1]]
        W = W[:,ind[::-1]]
    elif (which == "SR") or (which == "SA"):
        ind = np.argsort(Scopy)
        S = S[ind]
        W = W[:,ind]
    return S, W
        

def _mgsm(phi, Bphi, u):
    """Modificed M-orthogonal Gram-Schmidt"""
    for i in range(Bphi.shape[1]):
        u = u - np.dot(phi,np.dot(Bphi.T,u))
    return u
        
def _icgsm(phi, B, u):
    """Iterative classical M-orthogonal Gram-Schmidt"""
    alpha = 0.5; itmax = 3;
    Bu = B(u)
    r0 = np.sqrt(np.dot(u.T,Bu))
    for it in range(itmax):
        u = u - np.dot(phi,np.dot(phi.T,Bu))
        Bu = B(u)
        r1 = np.sqrt(np.dot(u.T,Bu))
        if r1 > alpha*r0:
            break
        r0 = r1
    if r1 < alpha*r0:
        warnings.warn("icgsm experienced loss of orthogonality")
    return u, r1

def gd(A, k=6, which="LM", B=None,
            M=None, X=None, Y=None,
            sigma=None, tol=1e-8, maxiter=100,
            multiple=1e-5, jmin=None, jmax=None,
            verbosityLevel=0, retLambdaHistory=False,
            retResidualNormsHistory=False, retPhiHistory=False,
            retUpdateHistory=False,
            Elements=None, Nodes=None, el_color=None, exact=None):
    """Solves the eigenvalue problem with generalized davidson method

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The symmetric linear operator of the problem, usually a
        sparse matrix.  Often called the "stiffness matrix".
    k : integer, optional
        The number of eigenvalues to solve for.  If <0, solves k eigenvalues,
        where k>m and w[-1] != w[-2].  This provides a way to ensure that all 
        multiples of the last eigenvalue are found, if desired.
    which : string, optional
        Type of eigenvalues to search for (e.g. LR or SM)
    B : {dense matrix, sparse matrix, LinearOperator}, optional
        the right hand side operator in a generalized eigenproblem.
        by default, B = Identity
        often called the "mass matrix"
    M : {dense matrix, sparse matrix, LinearOperator}, optional
        preconditioner to A; by default M = Identity.
    X : array_like, optional
        Initial search space.  Set to random of size jmin by default
    Y : array_like, optional
        n-by-sizeY matrix of constraints, sizeY < n
        The iterations will be performed in the B-orthogonal complement
        of the column-space of Y. Y must be full rank.
    sigma : scalar, optional
        Shift to apply when searching for eigenvalues.

    Returns
    -------
    lam : array
        Array of k eigenvalues
    phi : array
        An array of k eigenvectors.

    Other Parameters
    ----------------
    tol : scalar, optional
        Solver tolerance (stopping criterion)
        by default: tol=1e-8
    maxiter : integer, optional
        maximum number of iterations
        by default: maxiter=min(n,20)
    multiple: 0 < scalar < 1, optional
        For determining if eigenvalues are multiples, only used when k < 0
    verbosityLevel : integer, optional
        controls solver output.  default: verbosityLevel = 0.
    retLambdaHistory : boolean, optional
        whether to return eigenvalue history
    retResidualNormsHistory : boolean, optional
        whether to return history of residual norms
    retPhiHistory : boolean, optional
        whether to return eigenvector history
    retUpdateHistory : boolean, optional
        whether to return search space expansion history
    jmin, jmax : integer, optional
        Minimum and maximum size of search space, default is max(2*m,10) or max(4*m,15)
    Elements : array_like, optional
        Element connectivity for underlying mesh if intermediate plotting of
        approximated eigenmodes is desired
    Nodes : array_like, optional
        Node coordinates for underlying mesh if intermediate plotting of
        approximated eigenmodes is desired
    el_color : array_like, optional
        Element coloring for underlying mesh if intermediate plotting of
        approximated eigenmodes is desired

    Examples
    --------
    """
    
    # Some additional parsing of inputs where necessary
    n = A.shape[0]
    which = which.upper()
    if B is None:
        B = sparse.eye(n)
    if M is None:
        M = sparse.eye(n)
    if jmin is None:
        jmin = min( max(2*k,10), n/2 )
    if jmax is None:
        jmax = min( max(8*k,25), n )
    if X is None:
        X = np.random.rand(n,jmin)
        
    if k < 0:
        k *= -1
        ktype = "Distinct"
    else:
        ktype = "Total"
        
    A = _makeOperator(A, (n,n))
    B = _makeOperator(B, (n,n))
    M = _makeOperator(M, (n,n))

    X = _b_orthonormalize(B, X)[0]
    
    j = X.shape[1]
    m = 0
    phi = np.zeros((n,0))
    Bphi = np.zeros((n,0))
    phihat = np.zeros((n,1))
    Bphihat = np.zeros((n,1))
    lam = np.zeros(0)
    GA = np.dot(X.T, A(X))
    GB = np.dot(X.T, B(X))
    
    lambdaHistory = []
    residualNormsHistory = []
    phiHistory = []
    zHistory = []
    
    if Elements is not None and Nodes is not None:
        if verbosityLevel >= 2:
            print("Setting up mesh for plotting")
        patches, collection = _CreateCollection(Elements, Nodes, el_color)
        if verbosityLevel >= 2:
            print("Mesh ready for plotting")
    
    istep = 0
    
    for it in range(maxiter):
        # Solve Ritz problem
        S,W = eigh(GA, GB, check_finite=False)
        S,W = _sorteig(S,W,which,sigma)
        
        # Update eigenvector approximation
        u = np.dot(X,W[:,0])
        theta = S[0]
        Bu = B(u)
        Au = A(u)
        r = Au-theta*Bu
        rnorm = np.linalg.norm(r)
        phihat[:,-1] = u
        Bphihat[:,-1] = Bu
           
        if verbosityLevel >= 2:
            print("Iteration: %i, Converged: %i, lambda: %7.6g, residual: %7.6g"
                  %(it, m, theta, rnorm))
        lambdaHistory.append(theta)
        residualNormsHistory.append(rnorm)
        phiHistory.append(u)
        
        # Convergence check
        converged = residualNormsHistory[-1]/abs(theta) < tol
        morespace = j > 1
        if ktype == "Total":
            lastone = m==k-1
        else:
            distinct = abs(theta-lam[-1])/abs(lam[-1]) > multiple
            lastone = m>k and distinct
        if converged and (morespace or lastone):
            if verbosityLevel >= 1:
                print("Mode %i converged, lambda = %7.6g"%(m+1, theta))
            X = np.dot(X,W[:,1:])
            S = S[1:]
            GA = np.diag(S)
            GB = np.dot(X.T, B(X))
            W = np.identity(S.size)
            u = X[:,0]
            Bu = B(u)
            Au = A(u)
        
            lam = np.append(lam, theta)
            phi = phihat
            phihat = np.hstack([phi, u.reshape(-1,1)])
            Bphi = Bphihat
            Bphihat = np.hstack([Bphi, Bu.reshape(-1,1)])
            j-=1; m+=1; istep=1
            
            if lastone:
                ret = [lam, phi]
                if retLambdaHistory:
                    ret.append(lambdaHistory)
                if retResidualNormsHistory:
                    ret.append(residualNormsHistory)
                if retPhiHistory:
                    ret.append(phiHistory)
                if retUpdateHistory:
                    ret.append(zHistory)
                return ret
            
        if j >= jmax:
            # Restart
            j = jmin
            X = np.dot(X,W[:,:jmin])
            S = S[:jmin]
            GA = np.diag(S)
            GB = np.identity(jmin)
            
            
            # Optional Plotting
        if Elements is not None and Nodes is not None:
            if exact is not None and exact.shape[1] > m:
                sign = np.dot(exact[:,m], Bu)
                sign /= np.abs(sign)
                _Plot(Elements, Nodes, exact[:,m]-sign*u, el_color, patches, collection, 
                      "Mode %i error = %1.4g"%(m+1, np.linalg.norm(exact[:,m] - sign*u)),
                      "Progress2\\Iter_%i_Mode_%i.png"%(it, m+1))
            else:
                _Plot(Elements, Nodes, u, el_color, patches, collection, "Mode %i approximation"%(m+1))

        z = M(-r)
        z = _mgsm(phihat, Bphihat, z)
        z,r = _icgsm(X, B, z)
        z = z/r
        zHistory.append(z)

        Az = A(z)
        Bz = B(z)
        XAz = np.dot(X.T, Az).reshape(-1,1)
        XBz = np.dot(X.T, Bz).reshape(-1,1)
        GA = np.vstack([np.hstack([GA, XAz]),
                       np.hstack([XAz.T, np.dot(z.T,Az).reshape(1,1)])])
        GB = np.vstack([np.hstack([GB, XBz]),
                       np.hstack([XBz.T, np.dot(z.T,Bz).reshape(1,1)])])
        X = np.hstack([X, z.reshape(-1,1)])
        j+=1; istep+=1; it+=1;
        
    sys.stderr.write("Only %i eigenvalues converged\n"%m)
    S,W = eigh(GA, GB, check_finite=False)
    S,W = _sorteig(S,W,which,sigma)
    phihat = np.hstack([phi, np.dot(X,W[:,:3])])
    lambdaHistory.append(S[0])
    phiHistory.append(np.dot(X,W[:,0]))
    r = A(phiHistory[-1]) - S[0] * B(phiHistory[-1])
    residualNormsHistory.append(np.linalg.norm(r))
    
    lam = np.append(lam, theta)
    ret = [lam, phihat]
    if retLambdaHistory:
        ret.append(lambdaHistory)
    if retResidualNormsHistory:
        ret.append(residualNormsHistory)
    if retPhiHistory:
        ret.append(phiHistory)
    if retUpdateHistory:
        ret.append(zHistory)
    return ret
      
def _CreateCollection(Elements, Nodes, el_color):
    patches = []
    for el in range(Elements.shape[0]):
        polygon = Polygon(Nodes[Elements[el,:],:], True)
        patches.append(polygon)
    collection = UpdatablePatchCollection(patches, cmap=cmap.binary)
    collection.set_edgecolor('c')
        
    if el_color is not None:
        collection.set_array(el_color.reshape(-1))
        
    plt.figure("Progress", figsize=(10,10))
    plt.clf()
    ax = plt.gca()
    ax.add_collection(collection)
        
    return patches, collection
        
def _Plot(Elements, Nodes, u, el_color, patches, collection, title, filename=None):
    xsize = np.max(Nodes[:,0]) - np.min(Nodes[:,0])
    ysize = np.max(Nodes[:,1]) - np.min(Nodes[:,1])
    scale = min(0.1*xsize/np.max(np.abs(u[::2])), 0.1*ysize/np.max(np.abs(u[1::2])))
    DShape = Nodes + scale*u.reshape((Nodes.shape[0],-1))
    
    for el in range(Elements.shape[0]):
        patches[el].set_xy(DShape[Elements[el,:],:])
        
    plt.figure("Progress", figsize=(10,10))
    plt.axis('equal')
    plt.xlim(np.min(DShape[:,0]), np.max(DShape[:,0]))
    plt.ylim(np.min(DShape[:,1]), np.max(DShape[:,1]))
    plt.title(title + ", Scale = %1.4g"%scale)
    plt.draw()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.pause(0.05)