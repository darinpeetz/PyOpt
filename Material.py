# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:37:22 2019

@author: Darin
"""
import numpy as np
    
class PlaneStressElastic:
    """ Linear elastic plane stress material
    """
    
    def __init__(self, E, Nu):
        """Constructor 
        
        Parameters
        ----------
        E : scalar
            Young's Modulus
        Nu : scalar
            Poisson's ratio
        
        """
        self.D = E/(1-Nu**2) * np.array([[1, Nu, 0], [Nu, 1, 0], [0, 0, (1-Nu)/2]])
    
    def Get_D(self):
        """Returns the constitutive matrix 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        D : array_like
            Constitutive matrix
            
        """
        
        return self.D
    
class PlaneStrainElastic:
    """ Linear elastic plane strain material
    """
    
    def __init__(self, E, Nu):
        """Constructor 
        
        Parameters
        ----------
        E : scalar
            Young's Modulus
        Nu : scalar
            Poisson's ratio
        
        """
        self.D = (E/((1+Nu) * (1-2*Nu)) * np.array([[1-Nu, Nu, 0],
                                                    [Nu, 1-Nu, 0],
                                                    [0, 0, (1-2*Nu)/2]]))
    
    def Get_D(self):
        """Returns the constitutive matrix 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        D : array_like
            Constitutive matrix
            
        """
        
        return self.D
 
class Elastic3D:
    """ 3D Linear elastic material
    """
    
    def __init__(self, E, Nu):
        """Constructor 
        
        Parameters
        ----------
        E : scalar
            Young's Modulus
        Nu : scalar
            Poisson's ratio
        
        """
        c = E / ((1+Nu) * (1-2*Nu))
        G = E / (2 * (1+Nu))
        self.D = np.array([[(1-Nu)*c, Nu*c, Nu*c, 0, 0, 0],
                           [Nu*c, (1-Nu)*c, Nu*c, 0, 0, 0],
                           [Nu*c, Nu*c, (1-Nu)*c, 0, 0, 0],
                           [0, 0, 0, G, 0, 0],
                           [0, 0, 0, 0, G, 0],
                           [0, 0, 0, 0, 0, G]])
    
    def Get_D(self):
        """Returns the constitutive matrix 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        D : array_like
            Constitutive matrix
            
        """
        
        return self.D