# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:23:39 2019

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
import davidson as dvd
from time import time
import cvxopt; import cvxopt.cholmod

class it_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.it = 0
    def __call__(self, rk=None):
        self.it += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.it, str(rk)))
            
def Compliance(fem, matVals):
        """ Computes compliance, defined as the inner product of force and
        displacement
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        matVals : dict
            Interpolated material values and their sensitivities
        
        Returns
        -------
        obj : scalar
            Function value
        dobjdE : array_like
            Sensitivity of function with respect to element stiffnesses
        dobjdV : array_like
            Sensitivity of function with respect to element densities
        
        """
        
        obj = np.inner(fem.F, fem.U)
        elNDof = np.array([(fem.nDof*len(el))**2 for el in fem.elements])
        sumPoints = np.cumsum(elNDof)-1
        dCdE = np.cumsum(-fem.U[fem.i] * fem.k * fem.U[fem.j])[sumPoints]
        dCdE[1:] = dCdE[1:] - dCdE[:-1]
        
        return obj, dCdE * matVals['dEdy'], np.zeros(dCdE.shape)

def Volume(fem, matVals):
        """ Computes total volume fraction of the structure
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        matVals : dict
            Interpolated material values and their sensitivities
        
        Returns
        -------
        obj : scalar
            Function value
        dobjdE : array_like
            Sensitivity of function with respect to element stiffnesses
        dobjdV : array_like
            Sensitivity of function with respect to element densities
        
        """
        
        obj = np.inner(fem.areas, matVals['V']) / fem.domainSize
        
        return obj, np.zeros(fem.areas.shape), fem.areas / fem.domainSize * matVals['dVdy']

def SurfaceArea(fem, matVals):
        """ Computes total surface area/perimeter of the structure
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        matVals : dict
            Interpolated material values and their sensitivities
        
        Returns
        -------
        obj : scalar
            Function value
        dobjdE : array_like
            Sensitivity of function with respect to element stiffnesses
        dobjdV : array_like
            Sensitivity of function with respect to element densities
        
        """
        
        temp = np.append(matVals['V'], [0])
        jump = temp[fem.edgeElems[:,0]] - temp[fem.edgeElems[:,1]]
        obj = np.inner(np.abs(jump), fem.edgeLengths)
        
        dPdV = np.zeros(fem.nElem + 1)
        slope = fem.edgeLengths * np.sign(jump)
        for i in range(slope.size):
            dPdV[fem.edgeElems[i,0]] += slope[i]
            dPdV[fem.edgeElems[i,1]] -= slope[i]
            
        dPdV = dPdV[:-1]
        
        return -obj, np.zeros(dPdV.shape), -dPdV * matVals['dVdy']

def Stability(fem, matVals):
        """ Computes critical buckling mode of the structure
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        matVals : dict
            Interpolated material values and their sensitivities
        
        Returns
        -------
        obj : scalar
            Function value
        dobjdE : array_like
            Sensitivity of function with respect to element stiffnesses
        dobjdV : array_like
            Sensitivity of function with respect to element densities
        
        """
        nev = 6
        if not hasattr(fem, 'phi'):
            fem.phi = np.random.rand(fem.U.size, nev)
        
        Ks, dKsdy = _StressStiffness(fem, matVals, fem.U)
        
        springlessDof = np.setdiff1d(fem.freeDof, fem.springDof)
        
        interiorDiag = np.zeros(fem.K.shape[0])
        interiorDiag[springlessDof] = 1.
        interiorDiag = sparse.spdiags(interiorDiag, 0, fem.K.shape[0], fem.K.shape[1])
        exteriorDiag = np.ones(fem.K.shape[0])
        exteriorDiag[springlessDof] = 0.
        exteriorDiag = sparse.spdiags(exteriorDiag, 0, fem.K.shape[0], fem.K.shape[1])
        
        Ks = interiorDiag * Ks * interiorDiag + 1e-10*exteriorDiag
        fem.K = interiorDiag * fem.K * interiorDiag + exteriorDiag

        if hasattr(fem, 'ml_AMG'):
            # Use Davidson with AMG
            M = fem.ml_AMG.aspreconditioner()
            lam, phi = dvd.gd(Ks,B=fem.K,M=M,which="LA",maxiter=1000,k=nev,tol=1e-5,
                              verbosityLevel=0, X=fem.phi.copy())
        elif hasattr(fem, 'ml_GMG'):
            # Use Davidson with GMG
            M = fem.ml_GMG.aspreconditioner()
            lam, phi = dvd.gd(Ks,B=fem.K,M=M,which="LA",maxiter=1000,k=nev,tol=1e-5,
                              verbosityLevel=0, X=fem.phi.copy())
        elif hasattr(fem, 'ml_HYBRID'):
            # Use Davidson with hybrid MG
            M = fem.ml_HYBRID.aspreconditioner()
            lam, phi = dvd.gd(Ks,B=fem.K,M=M,which="LA",maxiter=1000,k=nev,tol=1e-5,
                              verbosityLevel=0, X=fem.phi.copy())
        else:
            springlessDof = np.setdiff1d(fem.freeDof, fem.springDof)
            K = fem.K.tocsr()
            K = K[springlessDof, :]
            K = K[:, springlessDof]
            K = K.tocoo()
            
            K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
            Kfact = cvxopt.cholmod.symbolic(K)
            cvxopt.cholmod.numeric(K, Kfact)
            
            # Use ARPACK
            def KInv(v):
                B = cvxopt.matrix(v[springlessDof])
                cvxopt.cholmod.solve(Kfact,B)
                b = 0*v
                b[springlessDof] = np.array(B).ravel()
                return b
            A = sparse.linalg.LinearOperator((fem.K.shape), matvec=KInv, rmatvec=KInv)
            lam, phi = sparse.linalg.eigsh(Ks, M=fem.K, which='LA', k=nev, Minv=A)
        
        index = np.argsort(lam)[::-1]
        lam = lam[index]
        phi = phi[:,index]
        
        fem.lamda = lam
        fem.phi = phi
        
        
        # Facilitate displacement sensitivity of stress stiffness matrix
        if not hasattr(fem, 'dKsdu'):
            fem.v = np.zeros((fem.U.size, nev))
            fem.dKsdu = []
            for el in range(fem.nElem):
                nDof = fem.dof[el].size
                dKsdu = np.zeros((nDof**2, nDof))
                for dof in range(nDof):
                    du = np.zeros(nDof)
                    du[dof] = 1
                    stress = np.dot(fem.DB[el], du)
                    dksdu = np.dot(fem.G[el].T, np.dot(_sigtos(stress), fem.G[el]))
                    for i in range(1, fem.nDof):
                        dksdu[i::fem.nDof,i::fem.nDof] = dksdu[::fem.nDof,::fem.nDof]
                    dKsdu[:,dof] = dksdu.ravel()
                fem.dKsdu.append(dKsdu)
                if fem.uniform:
                    break
                
        # Construct adjoint vectors
        dKsdU = np.zeros((fem.nodes.size, nev))
        ind = 0
        for el in range(fem.nElem):
            if fem.uniform:
                dKs = matVals['Es'][el] * fem.dKsdu[0]
            else:
                dKs = matVals['Es'][el] * fem.dKsdu[el]
            ind = np.arange(ind, ind + dKs.shape[0])
            phi_el = phi[fem.i[ind],:] * phi[fem.j[ind],:]
            dKsdU[fem.dof[el],:] += np.dot(dKs.T, phi_el)
            ind = ind[-1] + 1
            
        v = np.zeros_like(dKsdU)
        dKsdU[fem.fixDof] = 0
        
        if fem.v.shape[1] != nev:
            temp = np.zeros((fem.U.size, nev))
            temp[:,:fem.v.shape[1]] = fem.v
            fem.v = temp
        x0 = fem.v
        x0 = np.zeros((fem.U.size, nev))
        

        if hasattr(fem, 'ml_AMG'):
            M = fem.ml_AMG.aspreconditioner()
            for i in range(nev):
                v[:,i], info = sparse.linalg.cg(fem.ml_AMG.levels[0].A, dKsdU[:,i],
                                                tol=1e-05, M=M, x0=x0[:,i],
                                                maxiter=0.03*fem.K.shape[0])
        if hasattr(fem, 'ml_GMG'):
            M = fem.ml_GMG.aspreconditioner()
            for i in range(nev):
                v[:,i], info = sparse.linalg.cg(fem.ml_GMG.levels[0].A, dKsdU[:,i],
                                                tol=1e-05, M=M, x0=x0[:,i],
                                                maxiter=0.03*fem.K.shape[0])
        if hasattr(fem, 'ml_HYBRID'):
            M = fem.ml_HYBRID.aspreconditioner()
            for i in range(nev):
                v[:,i], info = sparse.linalg.cg(fem.ml_HYBRID.levels[0].A, dKsdU[:,i],
                                                tol=1e-05, M=M, x0=x0[:,i],
                                                maxiter=0.03*fem.K.shape[0])
        else:
            B = cvxopt.matrix(dKsdU[fem.freeDof,:])
            cvxopt.cholmod.solve(fem.Kfact,B)
            v[fem.freeDof,:] = np.array(B).reshape(fem.freeDof.size, -1)
            
        fem.v = v.copy()
        
        # Compile sensitivities
        dfdy = np.zeros((fem.nElem, nev))
        ind = 0
        for el in range(fem.nElem):
            if fem.uniform:
                ind = np.arange(ind, ind + fem.dKsdu[0].shape[0])
            else:
                ind = np.arange(ind, ind + fem.dKsdu[el].shape[0])
            dKdy = matVals['dEdy'][el] * fem.k[ind]
            ii = fem.i[ind]
            jj = fem.j[ind]
            dfdy[el,:] = ( np.sum(phi[ii,:] * phi[jj,:] *
                        (np.tile(dKsdy[ind].reshape(-1,1), (1, nev)) -
                         np.dot(dKdy.reshape(-1,1), lam.reshape(1,-1))), axis=0) +
                                np.dot(fem.U[jj]*dKdy, v[ii,:]) )
            ind = ind[-1] + 1
        
        fem.dlamda = dfdy.copy()
        p = 8
        f = np.sum(lam**p)**(1/p)
        df = f**(1-p) * np.sum(lam**(p-1)*dfdy, axis=1)
        return f, df, np.zeros(fem.nElem)
    
def _StressStiffness(fem, matVals, U):
        """ Assembles the stress stiffness matrix and it's design sensitivity
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        matVals : dict
            Interpolated material values and their sensitivities
        U : array_like
            Displacement vector
        
        Returns
        -------
        Ks : sparse matrix
            Stress stiffness matrix
        dKs : array_like
            Stress stiffness matrix sensitivity to design values
        """
        
        if not hasattr(fem, "dof"):
            offset = np.arange(fem.nDof).reshape(1,-1)
            fem.dof = [(fem.nDof*el.reshape(-1,1) + offset).ravel() for el in fem.elements]
        
        Ks = np.zeros_like(fem.i, dtype=float)
        dKs = np.zeros_like(fem.i, dtype=float)
        ind = 0
        for el in range(fem.nElem):
            if fem.uniform:
                stress = np.dot(fem.DB[0], U[fem.dof[el]])
                G = fem.G[0]
            else:
                stress = np.dot(fem.DB[el], U[fem.dof[el]])
                G = fem.G[el]
            ks = np.dot(G.T, np.dot(_sigtos(stress), G))
            for i in range(1, fem.nDof):
                ks[i::fem.nDof,i::fem.nDof] = ks[::fem.nDof,::fem.nDof]
            Ks[ind:ind+ks.size] = -matVals['Es'][el] * ks.ravel()
            dKs[ind:ind+ks.size] = -matVals['dEsdy'][el] * ks.ravel()
            ind += ks.size
            
        return sparse.bsr_matrix((Ks, (fem.i, fem.j)), blocksize=(fem.nDof, fem.nDof)), dKs
            
def _sigtos(sigma):
        """ Convert from vector to matrix representation of stress
        
        Parameters
        ----------
        sigma : array_like
            Vector representation of stress
        
        Returns
        -------
        s : array_like
            Matrix representation of stress
        """
        if sigma.size == 1:
            return sigma
        elif sigma.size == 12:
            temp = np.zeros(8)
            temp[::2] = sigma[2::3]
            sigma = sigma.reshape(-1,3)[:,:-1].ravel()
            return np.diag(temp[:-1],-1) + np.diag(sigma) + np.diag(temp[:-1], 1)
        elif sigma.size == 48:
            d1 = np.zeros(24)
            d1[::3] = sigma[3::6]
            d1[1::3] = sigma[4::6]
            d2 = np.zeros(24)
            d2[::3] = sigma[5::6]
            sigma = sigma.reshape(-1,6)[:,:-3].ravel()
            return (np.diag(d2[:-2], -2) + np.diag(d1[:-1], -1) +
                    np.diag(sigma) + np.diag(d1[:-1], 1) + np.diag(d2[:-2], 2))
        else:
            raise ValueError("Unexpected size of stress vector")
        
    