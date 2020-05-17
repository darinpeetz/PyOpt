# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:29:08 2019

@author: Darin
"""

import numpy as np

class Interpolation:
    """ Base class for interpolation, handles the maximum feature length aspect
    """
    def __init__(self, P, R, vdmin, p, q, minStiff=1e-10):
        """ Base class to handle max features
        
        Parameters
        ----------
        P : sparse matrix
            Density filter matrix
        R : sparse matrix
            Matrix used to sum element densities around each element
        vdmin : scalar
            Minimum volume of voids around each element
        p : scalar
            Penalty value for intermediate densities
        q : scalar
            Penalty value for overly thick features
        minStiff : scalar, optional
            Minimum Young's Modulus
            
        """
        self.P = P
        self.R = R
        self.vdmin = vdmin
        self.p = p
        self.q = q
        self.eps = minStiff
        
    def BaseInterpolate(self, x):
        """ Calculates element densities and a penalty where feature length
        is too large
        
        Parameters
        ----------
        x : array_like
            Design variables
            
        Returns
        -------
        y : array_like
            Sum of the voids around each element divided by minimum amount of
            voids (capped at 1) raise to penalty q
        z : array_like
            
        rho : array_like
        rhoq : array_like
        """
        # Minimum operation is to prevent numerical errors where rho > 1
        rho = np.minimum(self.P * x, 1)
        if self.q < 1:
            # No maximum feature restriction
            y = np.ones_like(rho)
            rhoq = np.zeros_like(rho)
            z = rho.copy()
        else:
            y = self.R * np.append((1-rho)**self.q, [1], axis=0)
            y = np.minimum(y / self.vdmin, 1)
            rhoq = self.q * (1-rho)**(self.q - 1) / self.vdmin
            z = rho * y
        
        return y, z, rho, rhoq
        
class SIMP(Interpolation):
    """ Standard SIMP interpolation
    """
    def __init__(self, P, R, vdmin, p, q, minStiff=1e-10):
        """ Creates the material interpolation object and stores the relevant
        parameters
        
        Parameters
        ----------
        P : sparse matrix
            Density filter matrix
        R : sparse matrix
            Matrix used to sum element densities around each element
        vdmin : scalar
            Minimum volume of voids around each element
        p : scalar
            Penalty value for intermediate densities
        q : scalar
            Penalty value for overly thick features
        minStiff : scalar, optional
            Minimum Young's Modulus
            
        """
        Interpolation.__init__(self, P, R, vdmin, p, q, minStiff)
        
    def Interpolate(self, x):
        """ Calculate the interpolated material values
        
        Parameters
        ----------
        x : array_like
            Design variables
        
        Returns
        -------
        matVals : dict
            Interpolated material values and their sensitivities
            
        """
        
        y, z, rho, rhoq = self.BaseInterpolate(x)
        
        matVals = {'y':y, 'rho':rho, 'rhoq':rhoq}
        matVals['E'] = (z>0) * self.eps + (1-self.eps) * z**self.p
        matVals['Es'] = z**self.p
        matVals['V'] = rho
        matVals['dEdy'] = (1-self.eps) * self.p * z**(self.p-1)
        matVals['dEsdy'] = self.p * z**(self.p-1)
        matVals['dVdy'] = np.ones(rho.shape)
        
        return matVals
        
class SIMP_CUT(Interpolation):
    """ Standard SIMP interpolation with a minimum density for stress-stiffness
    interpolation
    """
    def __init__(self, P, R, vdmin, p, q, cut, minStiff=1e-10):
        """ Creates the material interpolation object and stores the relevant
        parameters
        
        Parameters
        ----------
        P : sparse matrix
            Density filter matrix
        R : sparse matrix
            Matrix used to sum element densities around each element
        vdmin : scalar
            Minimum volume of voids around each element
        p : scalar
            Penalty value for intermediate densities
        q : scalar
            Penalty value for overly thick features
        cut : scalar
            Minimum density for stress stiffness to be nonzero
        minStiff : scalar, optional
            Minimum Young's Modulus
            
        """
        Interpolation.__init__(self, P, R, vdmin, p, q, minStiff)
        self.cut = cut
        
    def Interpolate(self, x):
        """ Calculate the interpolated material values
        
        Parameters
        ----------
        x : array_like
            Design variables
        
        Returns
        -------
        matVals : dict
            Interpolated material values and their sensitivities
            
        """
        
        y, z, rho, rhoq = self.BaseInterpolate(x)
        
        matVals = {'y':y, 'rho':rho, 'rhoq':rhoq}
        matVals['E'] = self.eps + (1-self.eps) * z**self.p
        matVals['Es'] = (z > self.cut) * z**self.p
        matVals['V'] = rho
        matVals['dEdy'] = (1-self.eps) * self.p * z**(self.p-1)
        matVals['dEsdy'] = self.p * (z > self.cut) * z**(self.p-1)
        matVals['dVdy'] = np.ones(rho.shape)
        
        return matVals
        
#class SIMP_LOGISTIC:
#    """ SIMP interpolation with a logistic function instead of a power law
#    """
#    def __init__(self, penal, rate, trans, minStiff=1e-10):
#        """ Creates the material interpolation object and stores the relevant
#        parameters
#        
#        Parameters
#        ----------
#        penal : scalar
#            Penalty value
#        rate : scalar
#            Controls the rate of transition of the logistic function
#        trans : scalar
#            Inflection point of logistic function
#        minStiff : scalar
#            Minimum Young's Modulus
#            
#        """
#        
#        self.p = penal
#        self.rate = rate
#        self.trans = trans
#        self.eps = minStiff
#        
#    def Interpolate(self, y):
#        """ Calculate the interpolated material values
#        
#        Parameters
#        ----------
#        y : array_like
#            Filtered material densities
#        
#        Returns
#        -------
#        matVals : dict
#            Interpolated material values and their sensitivities
#            
#        """
#        
#        matVals = {}
#        matVals['E'] = self.eps + (1-self.eps) * y**self.p
#        denom = 1 + np.exp(self.rate * (self.trans-y))
#        matVals['Es'] = (y**self.p) / denom
#        matVals['V'] = y
#        matVals['dEdy'] = (1-self.eps) * self.p * y**(self.p-1)
#        matVals['dEsdy'] = (self.p * y**(self.p-1) + (matVals['Es'] *
#                                    self.rate) * (1-1./denom)) / denom
#        matVals['dVdy'] = np.ones(y.shape)
#        
#        return matVals
#        
#class SIMP_SMOOTH:
#    """ SIMP interpolation with a smoothstep function instead of a power law
#    """
#    def __init__(self, penal, rate, trans, minStiff=1e-10):
#        """ Creates the material interpolation object and stores the relevant
#        parameters
#        
#        Parameters
#        ----------
#        penal : scalar
#            Penalty value
#        minimum : scalar
#            Point where the function hits zero
#        maximum : scalar
#            Point where the function hits one
#        minStiff : scalar
#            Minimum Young's Modulus
#            
#        """
#        
#        self.p = penal
#        self.minimum = minimum
#        self.maximum = maximum
#        self.eps = minStiff
#        
#    def Interpolate(self, y):
#        """ Calculate the interpolated material values
#        
#        Parameters
#        ----------
#        y : array_like
#            Filtered material densities
#        
#        Returns
#        -------
#        matVals : dict
#            Interpolated material values and their sensitivities
#            
#        """
#        
#        matVals = {}
#        matVals['E'] = self.eps + (1-self.eps) * y**self.p
#        matVals['Es'] = y**self.p
#        matVals['V'] = y
#        matVals['dEdy'] = (1-self.eps) * self.p * y**(self.p-1)
#        matVals['dEsdy'] = self.p * y**(self.p-1)
#        shift = (y-self.minimum) / (self.maximum-self.minimum)
#        adjust = ((6*shift**5 - 15*shift**4 + 10*shift**3) *
#                  (shift >= 0 & shift <= 1) + (shift > 1))
#        dadjust = ((30*shift**4 - 60*shift**3 + 30*shift**2) *
#                   (shift >= 0 & shift <= 1)/(self.maximum-self.minimum))
#        matVals['dEsdy'] *= adjust + matVals['Es'] * dadjust
#        matVals['Es'] *= adjust
#        matVals['dVdy'] = np.ones(y.shape)
#        
#        return matVals
#    