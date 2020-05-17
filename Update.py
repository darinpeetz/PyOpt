# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:51:26 2019

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
    

class OCUpdateScheme():
    """ The optimality criteria update scheme
    """
    
    def __init__(self, move, eta, x, xMin, xMax, passive=None):
        """ Update the design variables
        
        Parameters
        ----------
        move : scalar
            move limit
        eta : scalar
            power in the update scheme
        x : array_like
            Initial design values
        xMin : array_like
            Minimum value of each design variable
        xMax : array_like
            Maximum value of each design variable
        passive : array_like, optional
            Which elements will be passive
            
        """
        
        self.x = x
        self.active = np.arange(self.x.size)
        self.passive = np.array([], dtype=int)
        
        if passive is not None:
            self.active = np.setdiff1d(self.active, passive)
            self.passive = passive
            self.xMin = xMin[self.active]
            self.xMax = xMax[self.active]
        else:
            self.xMin = xMin
            self.xMax = xMax
            
        self.move = move * (self.xMax - self.xMin)
        self.eta = eta
        self.it = 0
        self.n = self.active.size
        self.m = 1
        
    def GetData(self):
        """ Get important data from the class
        
        Parameters
        ----------
        None
        
        Returns
        -------
        data : dict
            Dictionary of all important data in the structure
            
        """
        
        return {'x':self.x, 'it':self.it, 'xMin':self.xMin, 'xMax':self.xMax,
                'active':self.active, 'passive':self.passive,
                'move':self.move, 'eta':self.eta, 'type':'OC'}
        
    def Load(self, data):
        """ Rebuild the class with data from a file
        
        Parameters
        ----------
        data : dict
            Data from the file
        
        Returns
        -------
        None
            
        """
        
        self.x = data['x']
        self.active = data['active']
        self.passive = np.arange(len(self.x))
        self.passive = np.setdiff1d(self.passive, self.active)
        self.it = data['it']
        self.xMin = data['xMin']
        self.xMax = data['xMax']
        self.move = data['move']
        self.eta = data['eta']
        
    def LoadPetsc(self, folder, appendix=None, Endian='+'):
        """ Create Update structure from PETSc code results
        
        Parameters
        ----------
        folder : str
            folder containing all of the Petsc results
        appendix : str
            Appendix for result values to restart from, if none picks highest penalty
        Endian : char
            Indicates byte ordering ('=':default, '<':little Endian, '>':big Endian)
        
        Returns
        -------
        None
            
        """
        
        from os.path import sep
        from struct import unpack
        from PetscBinaryIO import PetscBinaryRead
        
        if appendix is None:
            from os import listdir
            try:
                appendix = '_pen' + max([float(file[5:-4]) for file in listdir(folder)
                                        if 'x_pen' in file and 'info' not in file])
            except:
                appendix = None
    
        if appendix is not None:
            self.x = PetscBinaryRead(folder + sep + "x%s.bin" % appendix)
            
        with open(folder + sep + "active.bin", mode='rb') as fh:
            data = fh.read()
        if data:
            self.active = np.where(np.array(unpack(Endian + len(data)*'?', data)))[0]
        self.passive = np.arange(len(self.x))
        self.passive = np.setdiff1d(self.passive, self.active)
        
    def Update(self, dfdx, g, dgdx):
        """ Update the design variables
        
        Parameters
        ----------
        dfdx : array_like
            Objective gradients
        g : scalar
            Constraint function value (<0 satisfied, >0 violated)
        dgdx : array_like
            Constraint gradients
            
        Returns
        -------
        Change : scalar
            Maximum change in the design variables
            
        """

        if hasattr(g, '__len__') and len(g) > 1:
            raise ValueError("OC update must not have multiple constraints")
        else:
            dfdx = dfdx[self.active]
            dgdx = dgdx[self.active].ravel()
            
        l1 = 0
        l2 = 1e6 
        x0 = self.x[self.active]
        while l2-l1 > 1e-4:
          lmid = (l1 + l2) / 2
          B = -(dfdx / dgdx) / lmid
          xCnd = self.xMin + (x0 - self.xMin) * B ** self.eta
          xNew = np.maximum(np.maximum(np.minimum(np.minimum(xCnd, x0 + self.move), 
                                       self.xMax), x0 - self.move), self.xMin)
          if (g+np.inner(dgdx, (xNew-x0))>0):
              l1=lmid
          else:
              l2=lmid
        
        change = np.max(np.abs(xNew - x0) / (self.xMax - self.xMin))
        self.x[self.active] = xNew
        self.it += 1
        return change
    
class MMA():
    """ The Method of Moving Asymptotes (Svanberg, 1987)
    """
    
    def __init__(self, x, m, xMin, xMax, maxit=100, passive=None,
                 subsolver='Dual', move=1.0):
        """ Update the design variables
        
        Parameters
        ----------
        x : array_like
            Initial design values
        m : integer
            Number of constraints
        xMin : array_like
            Minimum value of each design variable
        xMax : array_like
            Maximum value of each design variable
        maxit : integer
            Maximum number of subspace iterations
        passive : array_like, optional
            Which elements will be passive
        subsolver : string
            'Dual' or 'PrimalDual' to select which solver to use on the subproblem
        move : scalar
            Move limit for each design variable
            
        """
        
        self.x = x
        self.active = np.arange(self.x.size)
        self.passive = np.array([], dtype=int)
        
        if passive is not None:
            self.active = np.setdiff1d(self.active, passive)
            self.passive = passive
            self.xMin = xMin[self.active]
            self.xMax = xMax[self.active]
        else:
            self.xMin = xMin
            self.xMax = xMax
            
            
        self.xold1 = self.x[self.active]
        self.xold2 = self.x[self.active]
        self.n = self.active.size
        self.m = m
        self.maxit = maxit
        self.xRange = self.xMax - self.xMin
        
        self.d = np.ones(m)
        self.c = 1000 * np.ones(m)
        self.a0 = 1
        self.a = np.zeros(m)
        self.it = 0
        
        self.subsolver = subsolver
        self.move = move
        
    def GetData(self):
        """ Get important data from the class
        
        Parameters
        ----------
        None
        
        Returns
        -------
        data : dict
            Dictionary of all important data in the structure
            
        """
        
        return {'x':self.x, 'xold1':self.xold1, 'xold2':self.xold2, 'it':self.it,
                'xMin':self.xMin, 'xMax':self.xMax, 'a0':self.a0, 'a':self.a,
                'c':self.c, 'd':self.d, 'low':self.low, 'upp':self.upp,
                'active':self.active, 'passive':self.passive,
                'subsolver':self.subsolver, 'move':self.move, 'type':'MMA'}
        
    def Load(self, data):
        """ Rebuild the class with data from a file
        
        Parameters
        ----------
        data : dict
            Data from the file
        
        Returns
        -------
        None
            
        """
        
        self.x = data['x']
        self.it = data['it']
        self.active = data['active']
        self.passive = np.arange(len(self.x))
        self.passive = np.setdiff1d(self.passive, self.active)
        self.xold1 = data['xold1']
        self.xold2 = data['xold2']
        self.xMin = data['xMin']
        self.xMax = data['xMax']
        self.a0 = data['a0']
        self.a = data['a']
        self.c = data['c']
        self.d = data['d']
        self.low = data['low']
        self.upp = data['upp']
        
    def LoadPetsc(self, folder, appendix=None, Endian='+'):
        """ Create Update structure from PETSc code results
        
        Parameters
        ----------
        folder : str
            folder containing all of the Petsc results
        appendix : str
            Appendix for result values to restart from, if none picks highest penalty
        Endian : char
            Indicates byte ordering ('=':default, '<':little Endian, '>':big Endian)
        
        Returns
        -------
        None
            
        """
        
        from os.path import sep
        from struct import unpack
        from PetscBinaryIO import PetscBinaryRead
        
        if appendix is None:
            from os import listdir
            try:
                appendix = '_pen%g' % (max([float(file[5:-4]) for file in listdir(folder)
                                            if 'x_pen' in file and 'info' not in file]))
            except:
                appendix = None
    
        if appendix is not None:
            self.x = PetscBinaryRead(folder + sep + "x%s.bin" % appendix)
            
        with open(folder + sep + "active.bin", mode='rb') as fh:
            data = fh.read()
        if data:
            self.active = np.where(np.array(unpack(Endian + len(data)*'?', data)))[0]
            self.xMin = self.xMin[self.active]
            self.xMax = self.xMax[self.active]
            self.xRange = self.xMax - self.xMin
            self.n = self.active.size
            self.xold1 = self.x[self.active]
            self.xold2 = self.x[self.active]
            
        self.passive = np.arange(len(self.x))
        self.passive = np.setdiff1d(self.passive, self.active)
        
    def Update(self, dfdx, g, dgdx):
        """ Update the design variables
        
        Parameters
        ----------
        dfdx : array_like
            Objective gradients
        g : scalar
            Constraint function value (<0 satisfied, >0 violated)
        dgdx : array_like
            Constraint gradients
            
        Returns
        -------
        Change : scalar
            Maximum change in the design variables
            
        """
        
        albefa = 0.1
        asyinit = 0.5
        asyincr = 1.2
        asydecr = 0.7
        
        dfdx = dfdx[self.active]
        dgdx = dgdx[self.active]
        self.xact = self.x[self.active]
        
        # Calculation of the asymptotes low and upp
        if self.it < 2.5:
            self.low = self.xact - asyinit * self.xRange
            self.upp = self.xact + asyinit * self.xRange
        else:
            # Check for oscillations
            zzz = (self.xact - self.xold1) * (self.xold1 - self.xold2)
            factor = np.ones(self.n)
            factor[zzz > 0] = asyincr
            factor[zzz < 0] = asydecr
            self.low = self.xact - factor * (self.xold1 - self.low)
            self.upp = self.xact + factor * (self.upp - self.xold1)
            lowMin = self.xact - 10 * self.xRange
            lowMax = self.xact - 0.01 * self.xRange
            uppMin = self.xact + 0.01 * self.xRange
            uppMax = self.xact + 10 * self.xRange
            
            self.low = np.maximum(self.low, lowMin)
            self.low = np.minimum(self.low, lowMax)
            self.upp = np.minimum(self.upp, uppMax)
            self.upp = np.maximum(self.upp, uppMin)
            
        zzz1 = self.low + albefa * (self.xact - self.low)
        zzz2 = self.xact - self.move * self.xRange
        zzz = np.maximum(zzz1, zzz2)
        self.alfa = np.maximum(zzz, self.xMin)
        zzz1 = self.upp - albefa * (self.upp - self.xact)
        zzz2 = self.xact + self.move * self.xRange
        zzz = np.minimum (zzz1, zzz2)
        self.beta = np.minimum(zzz, self.xMax)
        
        # Calculating p0, q0, P, Q, and b
#        xmami = self.xMax - self.xMin
        xmami = self.upp - self.low
        xmamieps = 1e-5 * np.ones(self.n)
        xmami = np.maximum(xmami, xmamieps)
        xmamiinv = 1 / xmami
        ux1 = self.upp - self.xact
        ux2 = ux1 ** 2
        xl1 = self.xact - self.low
        xl2 = xl1 ** 2
        uxinv = 1 / ux1
        xlinv = 1 / xl1
        
        self.p0 = np.maximum(dfdx, 0)
        self.q0 = np.maximum(-dfdx, 0)
#        pq0 = 0.001 * (self.p0 + self.q0) + raa0 * xmamiinv
#        self.p0 += pq0
#        self.q0 += pq0
        self.p0 *= ux2
        self.q0 *= xl2
        
        self.pij = np.maximum(dgdx, 0)
        self.qij = np.maximum(-dgdx, 0)
#        self.pqij = 0.001 * (self.pij + self.qij) + raa0 * xmamiinv
#        self.pij += self.pqij
#        self.qij += self.pqij
        self.pij = (self.pij.T * ux2).T
        self.qij = (self.qij.T * xl2).T
                          
        self.b = np.dot(self.pij.T, uxinv) + np.dot(self.qij.T, xlinv) - g
        
        self.xold2 = self.xold1
        self.xold1 = self.xact
        
        if self.subsolver == 'Dual':
            self.DualSolve()
        elif self.subsolver == 'PrimalDual':
            self.PrimalSolve()
        else:
            raise ValueError('Subsolver type must be "Dual" or "PrimalDual"')
        
        change = np.max(np.abs(self.xact - self.xold1) / self.xRange)
        self.x[self.active] = self.xact
        self.it += 1
        return change
    
        
    def PrimalSolve(self):
        """ Solve the MMA sub-problem using dual method
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
            
        """
        
        epsimin = 1e-7
        epsi = 1.
        self.xact = 0.5*(self.alfa + self.beta)
        y = np.ones(self.m)
        z = 1.
        
        lamda = np.ones(self.m)
        xsi = 1 / (self.xact - self.alfa)
        xsi = np.maximum(xsi, 1)
        eta = 1 / (self.beta - self.xact)
        eta = np.maximum(eta, 1)
        mu  = np.maximum(1, 0.5*self.c)
        zet = 1
        s = np.ones(self.m)

        ux1, ux2, uxinv, xl1, xl2, xlinv, plam, qlam, gvec, dpsidx = \
                     self.PrimalTermsUpdate(lamda)
        
        while epsi > epsimin:            
            residunorm, residumax = self.PrimalResidual(epsi, lamda, y, z, s, gvec,
                                                        dpsidx, xsi, eta, mu, zet)[-2:]
            
            ittt = 0
            while residumax > 0.9 * epsi and ittt < self.maxit:
                ittt += 1
                
                ux1, ux2, uxinv, xl1, xl2, xlinv, plam, qlam, gvec, dpsidx = \
                     self.PrimalTermsUpdate(lamda)
                ux3 = ux1 * ux2
                xl3 = xl1 * xl2
                
                GG = self.pij.T * sparse.spdiags(1 / ux2, 0, self.n, self.n)
                GG -= self.qij.T * sparse.spdiags(1 / xl2, 0, self.n, self.n)
                
                delx = dpsidx - epsi / (self.xact - self.alfa) + epsi / (self.beta - self.xact)
                dely = self.c + self.d*y - lamda - epsi/y
                delz = self.a0 - np.inner(self.a, lamda) - epsi/z
                dellam = gvec - self.a*z - y - self.b + epsi / lamda
                
                diagx = plam/ux3 + qlam/xl3
                diagx = 2*diagx + xsi/(self.xact - self.alfa) + eta/(self.beta - self.xact)
                diagy = self.d + mu/y
                diaglam = s / lamda
                diaglamyi = diaglam + 1 / diagy
                
                if self.m < self.n:
                    blam = dellam + dely/diagy - np.dot(GG, delx/diagx)
                    bb = np.concatenate([blam, [delz]])
                    Alam = sparse.spdiags(diaglamyi, 0, self.m, self.m)
                    Alam += np.dot(GG, sparse.spdiags(1/diagx, 0, self.n, self.n) * GG.T)
                    AA = np.zeros((self.m+1, self.m+1))
                    AA[:self.m, :self.m] = Alam
                    AA[self.m:, :self.m] = self.a
                    AA[:self.m, self.m:] = self.a.reshape(-1,1)
                    AA[self.m:, self.m:] = -zet/z
                    
                    solut = np.linalg.solve(AA, bb)
                    dlam = solut[:self.m]
                    dz = solut[self.m]
                    dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
                    
                else:
                    diaglamyiinv = 1 / diaglamyi
                    dellamyi = dellam + dely / diagy
                    Axx = sparse.spdiags(diagx, 0, self.n, self.n)
                    Axx += np.dot(GG.T, sparse.spdiags(diaglamyiinv, 0, self.m, self.m) * GG)
                    azz = zet/z + np.inner(self.a, self.a / diaglamyi)
                    axz = -np.dot(GG.T, self.a / diaglamyi)
                    bx = delx + np.dot(GG.T, dellamyi / diaglamyi)
                    bz = delz - np.inner(self.a, dellamyi / diaglamyi)
                    AA = np.zeros((self.n+1, self.n+1))
                    AA[:self.n, :self.n] = Alam
                    AA[self.n:, :self.n] = axz
                    AA[:self.n, self.n:] = axz.reshape(-1,1)
                    AA[self.n:, self.n:] = azz
                    bb = np.concatenate([-bx, [-bz]])
                    
                    solut = np.linalg.solve(AA, bb)
                    dx = solut[:self.n]
                    dz = solut[self.n]
                    dlam = np.dot(GG, dx) / diaglamyi - dz*(self.a / diaglamyi)
                    dlam += dellamyi / diaglamyi
                    
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsi / (self.xact - self.alfa) - (xsi*dx) / (self.xact - self.alfa)
                deta = -eta + epsi / (self.beta - self.xact) + (eta*dx) / (self.beta - self.xact)
                dmu = -mu + epsi / y - (mu*dy) / y
                dzet = -zet + epsi/z - zet*dz/z
                ds   = -s + epsi/lamda - (s*dlam) /lamda
                xx  = np.concatenate([ y,  [z], lamda,  xsi,  eta,  mu,  [zet], s])
                dxx = np.concatenate([dy, [dz],  dlam, dxsi, deta, dmu, [dzet], ds])
                
                stepxx = -1.01 * dxx / xx
                stmxx = stepxx.max()
                stepalfa = -1.01*dx / (self.xact - self.alfa)
                stmalfa = stepalfa.max()
                stepbeta = 1.01 * dx / (self.beta - self.xact)
                stmbeta = stepbeta.max()
                stmalbe = max(stmalfa, stmbeta)
                stmalbexx = max(stmalbe, stmxx)
                stminv = max(stmalbexx, 1)
                steg = 1 / stminv
                
                xold   =   self.xact
                yold   =   y
                zold   =   z
                lamold =  lamda
                xsiold =  xsi
                etaold =  eta
                muold  =  mu
                zetold =  zet
                sold   =   s
                
                itto = 0
                resinew = 2*residunorm
                while resinew > residunorm and itto < 50:
                    itto += 1
                    self.xact =   xold + steg*dx
                    y         =   yold + steg*dy
                    z         =   zold + steg*dz
                    lamda     = lamold + steg*dlam
                    xsi       = xsiold + steg*dxsi
                    eta       = etaold + steg*deta
                    mu        = muold  + steg*dmu
                    zet       = zetold + steg*dzet
                    s         =   sold + steg*ds
                    
                    ux1, ux2, uxinv, xl1, xl2, xlinv, plam, qlam, gvec, dpsidx = \
                             self.PrimalTermsUpdate(lamda)
                    
                    resinew, residumax = self.PrimalResidual(epsi, lamda, y, z, s, gvec,
                                                             dpsidx, xsi, eta, mu, zet)[-2:]
                    steg /= 2
                    
                residunorm = resinew
                steg *= 2
                if ittt == self.maxit:
                    print('Maximum iterations reached for epsi = %1.2g' % epsi)
                    
            epsi *= 0.1
            
    def PrimalResidual(self, epsi, lamda, y, z, s, gvec, dpsidx, xsi, eta, mu, zet):
        """ Residual of primal subproblem
            
        Returns
        -------
        res : array_like
            Residual in the dual variables
        norm2 : scalar
            2-norm of res
        norminf : scalar
            Infinity-norm of res
            
        """
        
        rex = dpsidx - xsi + eta  # d/dx
        rey = self.c + self.d*y - mu - lamda  # d/dy
        rez = self.a0 - zet - np.inner(self.a, lamda)  # d/dz
        relam = gvec - self.a*z - y + s - self.b  # d/dlam
        rexsi = xsi * (self.xact - self.alfa) - epsi  # d/dxsi
        reeta = eta * (self.beta - self.xact) - epsi  # d/deta
        remu = mu * y - epsi  # d/dmu
        rezet = zet*z - epsi  # d/dzeta
        res = lamda * s - epsi  # d/ds
        
        residu1 = np.concatenate([rex, rey, [rez]])
        residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
        residual = np.concatenate([residu1, residu2])
        residunorm = np.linalg.norm(residual)
        residumax = np.max(np.abs(residual))
        
        return residual, residunorm, residumax
    
    def PrimalTermsUpdate(self, lamda):
        """ Update terms in the primal solver
        
        Parameters
        ----------
        None
            
        Returns
        -------
        ux1, ux2, uxinv, xl1, xl2, xlinv, plam, qlam, gvec, dpsidx
            
        """
        
        ux1 = self.upp - self.xact
        ux2 = ux1 ** 2
        uxinv = 1 / ux1
        xl1 = self.xact - self.low
        xl2 = xl1 ** 2
        xlinv = 1 / xl1
        
        plam = self.p0 + np.dot(self.pij, lamda)
        qlam = self.q0 + np.dot(self.qij, lamda)
        gvec = np.dot(self.pij.T, uxinv) + np.dot(self.qij.T, xlinv)
        dpsidx = plam/ux2 - qlam/xl2
        
        return ux1, ux2, uxinv, xl1, xl2, xlinv, plam, qlam, gvec, dpsidx
        
    def DualSolve(self):
        """ Solve the MMA sub-problem using dual method
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
            
        """
        
        epsimin = 1e-7
        epsi = 1.
        eta = np.ones(self.m)
        lamda = 500 * eta
        plam = self.p0 + np.dot(self.pij, lamda)
        qlam = self.q0 + np.dot(self.qij, lamda)
        y, z = self.XYZofLam(lamda, plam, qlam)
        
        ux1 = self.upp - self.xact
        ux2 = ux1 ** 2
        ux3 = ux1 * ux2
        xl1 = self.xact - self.low
        xl2 = xl1 ** 2
        xl3 = xl1 * xl2
        
        hvec = self.DualGradient(ux1, xl1, y, z)
        
        while epsi > epsimin:
            epsvecm = epsi * np.ones(self.m)
            
            residumax = self.DualResidual(hvec, eta, lamda, epsvecm)[-1]
            
            ittt = 0
            while residumax > 0.9 * epsi and ittt < self.maxit:
                ittt += 1
                
                ddpsi = self.grad2psi(lamda, ux2, xl2, ux3, xl3, plam, qlam)
                
                dellam, deleta = self.searchdir(lamda, eta, ddpsi, hvec, epsi)
                theta = self.searchdis(lamda, eta, dellam, deleta)
                
                lamda += theta * dellam
                eta += theta * deleta
                
                plam = self.p0 + np.dot(self.pij, lamda)
                qlam = self.q0 + np.dot(self.qij, lamda)
                y, z = self.XYZofLam(lamda, plam, qlam)
                
                ux1 = self.upp - self.xact
                ux2 = ux1 ** 2
                ux3 = ux1 * ux2
                xl1 = self.xact - self.low
                xl2 = xl1 ** 2
                xl3 = xl1 * xl2
                
                hvec = self.DualGradient(ux1, xl1, y, z)
                
                residumax = self.DualResidual(hvec, eta, lamda, epsvecm)[-1]
                
            if ittt == self.maxit:
                print('Maximum iterations reach for epsi = %1.2g' % epsi)
                
            epsi *= 0.1
            
        
    def DualResidual(self, hvec, eta, lamda, epsvecm):
        """ Residual of dual subproblem
        
        Parameters
        ----------
        hvec : array_like
            Gradients of the dual variables
        eta : array_like
        lamda : array_like
            Dual variables
        epsvecm : array_like
            
        Returns
        -------
        res : array_like
            Residual in the dual variables
        norm2 : scalar
            2-norm of res
        norminf : scalar
            Infinity-norm of res
            
        """
        
        reslam = hvec + eta
        reseta = eta * lamda - epsvecm
        res = np.concatenate([reslam.ravel(), reseta.ravel()])
        norm2 = np.linalg.norm(res, 2)
        norminf = np.abs(res).max()
        
        return res, norm2, norminf

        
    def XYZofLam(self, lamda, plam, qlam):
        """ Residual of dual subproblem
        
        Parameters
        ----------
        lamda : array_like
            Dual variables
        plam : array_like
        qlam : array_like
            
        Returns
        -------
        y : array_like
        z : scalar
            
        """
        
        plamrt = np.sqrt(plam)
        qlamrt = np.sqrt(qlam)
        self.xact = (plamrt * self.low + qlamrt * self.upp) / (plamrt + qlamrt)
        self.xact = np.maximum(np.minimum(self.xact, self.beta), self.alfa)
        
        y = np.maximum((lamda - self.c) / self.d, 0)
        
        z = 10 * max(np.inner(lamda, self.a) - self.a0, 0)
        
        return y, z
    
    def DualGradient(self, ux1, xl1, y, z):
        """ Residual of dual subproblem
        
        Parameters
        ----------
        ux1 : array_like
            upper asymptote minus x
        xl1 : array_like
            x minus lower asymptote
        y : array_like
        z : scalar
            
        Returns
        -------
        hvec : array_like
            Gradient of dual variables
            
        """
        
        hvec = np.sum(self.pij.T / ux1, axis=1) + np.sum(self.qij.T / xl1, axis=1)
            
        hvec -= self.b + self.a * z + y
        return hvec
    
    def grad2psi(self, lamda, ux2, xl2, ux3, xl3, plam, qlam):
        """ Computes Hessian of dual variables
        
        Parameters
        ----------
        lamda : array_like
            Dual variables
        ux2 : array_like
            upper asymptote minus x squared
        xl2 : array_like
            x minus lower asymptote squared
        ux2 : array_like
            upper asymptote minus x cubed
        xl2 : array_like
            x minus lower asymptote cubed
        plam : array_like
        qlam : array_like
            
        Returns
        -------
        ddpsi : array_like
            Hessian of dual variables
            
        """
        
        dhdx = (self.pij.T / ux2 - self.qij.T / xl2).T
            
        dLdxx = sparse.diags( np.logical_and(self.xact > self.alfa, self.xact < self.beta) /
                              (2 * plam / ux3 + 2 * qlam / xl3), 0)
        
        ddpsi = -np.dot(dhdx.T, dLdxx * dhdx)
        ddpsidy = lamda > self.c
            
        ddpsi -= np.diagonal(ddpsidy.reshape(-1,1))
        
        if np.inner(lamda, self.a) > 0:
            ddpsi -= 10 * np.inner(self.a, self.a)
            
        return ddpsi
            
    def searchdir(self, lamda, eta, ddpsi, hvec, epsi):
        """ Computes Hessian of dual variables
        
        Parameters
        ----------
        lamda : array_like
            Dual variables
        eta : array_like
        ddpsi : array_like
            Hessian of dual variables
        hvec : array_like
            Gradient of dual variables
        epsi : scalar
            Tolerance
            
        Returns
        -------
        dellam : array_like
            Search direction for lamda
        deleta : array_like
            Search direction for eta
            
        """

        A = ddpsi - sparse.diags(eta / lamda, 0)
        A += min(1e-4 * np.trace(A) / self.m, -1e-7) * sparse.identity(self.m)
        b = -hvec - epsi / lamda
        dellam = np.linalg.solve(A, b)
        deleta = -eta + epsi / lamda - dellam * eta / lamda
       
        return dellam, deleta

    def searchdis(self, lamda, eta, dellam, deleta):
        """ Computes Hessian of dual variables
        
        Parameters
        ----------
        lamda : array_like
            Dual variables
        eta : array_like
        dellam : array_like
            Search direction for lamda
        deleta : array_like
            Search direction for eta
            
        Returns
        -------
        theta : scalar
            Step size
            
        """
        
        ratio = -0.99 * lamda / dellam
        ratio[ratio < 0] = 1
        theta = min(ratio.min(), 1)
        ratio = -0.99 * eta / deleta
        ratio[ratio < 0] = 1
        theta = min(ratio.min(), theta)
        
        return theta
        