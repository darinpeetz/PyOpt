# -*- coding: utf-8 -*-
"""
Created on Wed May 08 10:39:48 2019

@author: Darin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Shape_Functions
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import Material
import pyamg
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
            
class FEM:
    """Provides functionality to solve the beam QC problem
    """
            
    def __init__(self):
        """Create a 1-element rectangular mesh by default
        
        Parameters
        ----------
        None
        
        Notes
        -----
        The proper calling order of functions is
        1 - CreateRecMesh
        2 - AddBc, AddLoad, and AddSpring; in any order
        3 - SetMaterial
        4 - Initialize
        5 - ConstructSystem
        6 - SolveSystem
        An example of this process is at the end of the file
        
        """
        self.elements = np.array([[0, 1, 3, 2]])
        self.nElem = 1
        self.nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.nNode, self.nDof = self.nodes.shape
        self.edgeElems = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.edgeLengths = np.ones(4)
        self.areas = np.array([1])
        
        self.fixDof = np.array([0, 1, 3], dtype=int)
        self.U = np.zeros(self.nodes.size)
        self.freeDof = np.array([2, 4, 5, 6, 7], dtype=int)
        self.F = np.zeros(self.nNode * self.nDof)
        self.F[5::2] = 1
        self.springDof = np.array([], dtype=int)
        self.stiff = np.array([])
    
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
        
        data = {'elements':self.elements, 'nodes':self.nodes, 'freeDof':self.freeDof,
                'fixDof':self.fixDof, 'U':self.U, 'F':self.F, 'areas':self.areas, 
                'springDof':self.springDof, 'stiff':self.stiff, 'P':self.P, 'uniform':self.uniform}
        
        return data
    
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
        
        self.elements = data['elements']
        self.nodes = data['nodes']
        self.freeDof = data['freeDof']
        self.fixDof = data['fixDof']
        self.F = data['F']
        self.springDof = data['springDof']
        self.stiff = data['stiff']
        if 'U' in data:
            self.U = data['U']
        else:
            self.U = np.zeros_like(self.F)
        self.areas = data['areas'] 
        self.domainSize = self.areas.sum()
        self.springDof = data['springDof']
        self.P = data['P']
        self.uniform = data['uniform']
        self.nElem = self.elements.shape[0]
        self.nNode, self.nDof = self.nodes.shape
        
        if 'k' in data:
            self.e = data['e']
            self.i = data['i']
            self.j = data['j']
            self.k = data['k']
            self.DB = data['DB']
            self.G = data['G']
            
    def LoadPetsc(self, folder, Endian='='):
        """ Create FEM structure from PETSc code results
        
        Parameters
        ----------
        folder : str
            folder containing all of the Petsc results
        Endian : char
            Indicates byte ordering ('=':default, '<':little Endian, '>':big Endian)
        
        Returns
        -------
        None
            
        """
        
        from os.path import sep
        from struct import unpack
        from PetscBinaryIO import PetscBinaryRead
        
        with open(folder + sep + "Element_Distribution.bin", mode='rb') as fh:
            data = fh.read()
        nProcessors = len(data)//4 # int size is 4 bytes
        dist = np.array(unpack(Endian + nProcessors*'i', data))
        self.nElem = dist[-1]
            
        with open(folder + sep + "elements.bin", mode='rb') as fh:
            data = fh.read()
        self.elements = np.array(unpack('<' + len(data)//4*'i', data)).reshape(self.nElem, -1)
        
        self.nNode = self.elements.max()+1
        with open(folder + sep + "nodes.bin", mode='rb') as fh:
            data = fh.read()
        self.nodes = np.array(unpack(Endian + len(data)//8*'d', data)).reshape(self.nNode, -1)
        self.nDof = self.nodes.shape[1]
        
        self.U = np.zeros(self.nNode*self.nDof, dtype=float)
        self.F = self.U.copy()
        
        # Fix degrees of freedom
        with open(folder + sep + "supportNodes.bin", mode='rb') as fh:
            data = fh.read()
        nodes = np.array(unpack(Endian + len(data)//4*'i', data))
        with open(folder + sep + "supports.bin", mode='rb') as fh:
            data = fh.read()
        conditions = np.array(unpack(Endian + len(data)*'?', data)).reshape(nodes.size, -1)
        for i in range(nodes.size):
            self.U[self.nDof*nodes[i]:self.nDof*(nodes[i]+1)] = conditions[i]
        self.fixDof = np.where(self.U > 0.5)[0]
        self.freeDof = np.where(self.U < 0.5)[0]
        self.U.fill(0)
        
        # Apply loads
        with open(folder + sep + "loadNodes.bin", mode='rb') as fh:
            data = fh.read()
        nodes = np.array(unpack(Endian + len(data)//4*'i', data))
        with open(folder + sep + "loads.bin", mode='rb') as fh:
            data = fh.read()
        loads = np.array(unpack(Endian + len(data)//8*'d', data)).reshape(nodes.size, -1)
        for i in range(nodes.size):
            self.F[self.nDof*nodes[i]:self.nDof*(nodes[i]+1)] = loads[i]
            
        # Apply springs
        with open(folder + sep + "springNodes.bin", mode='rb') as fh:
            data = fh.read()
        if len(data) == 0: # No springs
            self.springDof = []
            self.stiff = []
        else:
            nodes = np.array(unpack(Endian + len(data)//4*'i', data))
            with open(folder + sep + "springs.bin", mode='rb') as fh:
                data = fh.read()
            self.stiff = np.array(unpack(Endian + len(data)//8*'d', data))
            self.springDof = np.tile(nodes, (1, self.nDof)) + np.arange(self.nDof)
            self.springDof = self.springDof.ravel()
        
        vertices = self.nodes[self.elements]
        self.areas = np.sum(vertices[:,:,0] * np.roll(vertices[:,:,1], -1, axis=1) - 
                 vertices[:,:,0] * np.roll(vertices[:,:,1],  1, axis=1), axis=1) / 2
        self.domainSize = self.areas.sum()
        self.uniform = True
        
        self.P = []
        for level in range(20):
            try:
                self.P.append(PetscBinaryRead(folder + sep + "P%i.bin" % level).tobsr(
                                            blocksize=(self.nDof,self.nDof)))
            except:
                break
        
    # region basic creation methods
    def Create2DMesh(self, Dimensions, Nelx, Nely, maxCoarse=100, maxLevels=5):
        """ Creates a uniform rectangular finite element mesh structure
    
        Parameters
        ----------
        Dimensions : list
            [left edge, right edge, bottom edge, top edge]
        Nelx : integer
            Number of elements in the x direction
        Nely : integer
            Number of elements in the y direction
        maxCoarse : integer
            Maximum nodes on the coarse level (for use in GMG)
        maxLevels : integer
            Maximum number of MG levels (for use in GMG)
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the elements, nodes, edgeElems, and edgeLengths arrays to the FEM structure
        Also adds the flag uniform to indicate the mesh has uniform elements
    
        Examples
        --------
        fem = FEM()
        fem.CreateRecMesh([0, 1, 0, 1], 10, 10)

        """
        
        self.nElem = Nelx * Nely
        self.elements = np.zeros((self.nElem, 4), dtype=int)
        for ely in range(Nely):
            el = np.arange(ely*Nelx, (ely+1)*Nelx)
            offset = ely * (Nelx+1)
            self.elements[el,0] = np.arange(Nelx) + offset
            self.elements[el,1] = self.elements[el,0] + 1
            self.elements[el,2] = self.elements[el,1] + Nelx + 1
            self.elements[el,3] = self.elements[el,2] - 1
            
        xnodes, ynodes = np.meshgrid(np.linspace(Dimensions[0], Dimensions[1], Nelx+1),
                                     np.linspace(Dimensions[2], Dimensions[3], Nely+1))
        self.nodes = np.hstack([xnodes.reshape(-1,1), ynodes.reshape(-1,1)])
        self.nNode, self.nDof = self.nodes.shape
        self.U = np.zeros(self.nodes.size, dtype=float)
        self.F = self.U.copy()
        self.fixDof = np.array([], dtype=int)
        self.freeDof = np.arange(self.U.size, dtype=int)
        self.springDof = np.array([], dtype=int)
        self.stiff = np.array([], dtype=float)
        
        vertices = self.nodes[self.elements]
        self.areas = np.sum(vertices[:,:,0] * np.roll(vertices[:,:,1], -1, axis=1) - 
                 vertices[:,:,0] * np.roll(vertices[:,:,1],  1, axis=1), axis=1) / 2
        self.domainSize = self.areas.sum()
        self.uniform = True
        
        self.maxLevels = maxLevels
        self.maxCoarse = maxCoarse
        self.SetupGMGInterpolaters([Nelx+1, Nely+1], maxCoarse=maxCoarse, maxLevels=maxLevels)
        
    def Create3DMesh(self, Dimensions, Nelx, Nely, Nelz, maxCoarse=100, maxLevels=5):
        """ Creates a uniform brick finite element mesh structure
    
        Parameters
        ----------
        Dimensions : list
            [x0, x1, y0, y1, z0, z1]
        Nelx : integer
            Number of elements in the x direction
        Nely : integer
            Number of elements in the y direction
        Nelz : integer
            Number of elements in the z direction
        maxCoarse : integer
            Maximum nodes on the coarse level (for use in GMG)
        maxLevels : integer
            Maximum number of MG levels (for use in GMG)
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the elements, nodes, edgeElems, and edgeLengths arrays to the FEM structure
        Also adds the flag uniform to indicate the mesh has uniform elements
    
        Examples
        --------
        fem = FEM()
        fem.CreateRecMesh([0, 1, 0, 1], 10, 10)

        """
        
        self.nElem = Nelx * Nely * Nelz
        self.elements = np.zeros((self.nElem, 8), dtype=int)
        for elz in range(Nelz):
            for ely in range(Nely):
                el = np.arange(ely*Nelx + elz*Nelx*Nely, (ely+1)*Nelx + elz*Nelx*Nely)
                offset = ely * (Nelx+1) + elz*(Nelx+1)*(Nely+1)
                self.elements[el,0] = np.arange(Nelx) + offset
                self.elements[el,1] = self.elements[el,0] + 1
                self.elements[el,2] = self.elements[el,0] + Nelx + 2
                self.elements[el,3] = self.elements[el,0] + Nelx + 1
                self.elements[el,4] = self.elements[el,0] + (Nelx+1)*(Nely+1)
                self.elements[el,5] = self.elements[el,1] + (Nelx+1)*(Nely+1)
                self.elements[el,6] = self.elements[el,2] + (Nelx+1)*(Nely+1)
                self.elements[el,7] = self.elements[el,3] + (Nelx+1)*(Nely+1)
                
            
        xnodes = np.linspace(Dimensions[0], Dimensions[1], Nelx+1)
        ynodes = np.linspace(Dimensions[2], Dimensions[3], Nely+1)
        znodes = np.linspace(Dimensions[4], Dimensions[5], Nelz+1)
        xnodes = np.tile(xnodes, (Nely+1)*(Nelz+1))
        ynodes = np.tile(np.tile(ynodes.reshape(-1,1), Nelx+1).ravel(), Nelz+1)
        znodes = np.tile(znodes.reshape(-1,1), (Nelx+1)*(Nely+1)).ravel()
        self.nodes = np.hstack([xnodes.reshape(-1,1), ynodes.reshape(-1,1), znodes.reshape(-1,1)])
        self.nNode, self.nDof = self.nodes.shape
        self.U = np.zeros(self.nodes.size, dtype=float)
        self.F = self.U.copy()
        self.fixDof = np.array([], dtype=int)
        self.freeDof = np.arange(self.nodes.size, dtype=int)
        self.springDof = np.array([], dtype=int)
        self.stiff = np.array([], dtype=float)
        
        vertices = self.nodes[self.elements]
        self.areas = np.sum(vertices[:,:,0] * np.roll(vertices[:,:,1], -1, axis=1) - 
                 vertices[:,:,0] * np.roll(vertices[:,:,1],  1, axis=1), axis=1) / 2
        self.domainSize = self.areas.sum()
        self.uniform = True
        
        self.maxLevels = maxLevels
        self.maxCoarse = maxCoarse
        self.SetupGMGInterpolaters([Nelx+1, Nely+1, Nelz+1], maxCoarse=maxCoarse, maxLevels=maxLevels)
            
    def AddBC(self, bcSpecs):
        """ Adds boundary condition info to the FEM structure
    
        Parameters
        ----------
        bcSpecs : list
            Each entry is a dict with entry 'poly' describing a convex polygon
            that encompasses all nodes to have displacements specified in 'disp'
            applied to them. The poly entry is a list of nodal coordinates
            specifying vertices of the polygon. 'disp' should be a list/array
            with entries equal to the number of dof per node.
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the disp, fixDof, and freeDof arrays to the FEM structure

        """
        
        disp = np.empty(self.nNode * self.nDof)
        self.U = np.zeros(self.nNode * self.nDof)
        disp.fill(np.nan)
        for bcSpec in bcSpecs:
            if 'nodes' in bcSpec:
                inside = bcSpec['nodes']
            else:
                inside = np.where(np.logical_and(np.all(self.nodes > bcSpec['lower'], axis=1),
                                                 np.all(self.nodes < bcSpec['upper'], axis=1)))[0]
            for i in range(self.nDof):
                disp[self.nDof * inside + i] = bcSpec['disp'][i]
                
        self.fixDof = np.where(disp == disp)[0]
        self.freeDof = np.where(disp != disp)[0]
        self.U[self.fixDof] = disp[self.fixDof]
        
    def AddLoad(self, loadSpecs):
        """ Adds external loads info to the FEM structure
    
        Parameters
        ----------
        loadSpecs : list
            Each entry is a dict with entry 'poly' describing a convex polygon
            that encompasses all nodes to have displacements specified in 'disp'
            applied to them. The poly entry is a list of nodal coordinates
            specifying vertices of the polygon. 'force' should be a list/array
            with entries equal to the number of dof per node.
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the force vector to the FEM structure

        """
        
        self.F = np.zeros(self.nNode * self.nDof)
        for loadSpec in loadSpecs:
            if 'nodes' in loadSpec:
                inside = loadSpec['nodes']
            else:
                inside = np.where(np.logical_and(np.all(self.nodes > loadSpec['lower'], axis=1),
                                                 np.all(self.nodes < loadSpec['upper'], axis=1)))[0]
            for i in range(self.nDof):
                self.F[self.nDof * inside + i] += loadSpec['force'][i]
        
    def AddSprings(self, springSpecs):
        """ Adds springs to nodesin the FEM structure
    
        Parameters
        ----------
        springSpecs : list
            Each entry is a dict with entry 'poly' describing a convex polygon
            that encompasses all nodes to have displacements specified in 'disp'
            applied to them. The poly entry is a list of nodal coordinates
            specifying vertices of the polygon. 'stiff' should be a list/array
            with entries equal to the number of dof per node.
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the force vector to the FEM structure

        """
        
        self.stiff = np.empty(self.nNode * self.nDof)
        self.stiff.fill(np.nan)
        for springSpec in springSpecs:
            if 'nodes' in springSpec:
                inside = springSpec['nodes']
            else:
                inside = np.where(np.logical_and(np.all(self.nodes > springSpec['lower'], axis=1),
                                                 np.all(self.nodes < springSpec['upper'], axis=1)))[0]
            for i in range(self.nDof):
                self.stiff[self.nDof * inside + i] = springSpec['stiff'][i]
                
        self.springDof = np.where(self.stiff == self.stiff)[0]
        self.stiff = self.stiff[self.springDof]
        
    def SetMaterial(self, material):
        """ Adds a material object to the FEM class
    
        Parameters
        ----------
        material : Material object
            Defines the underlying material behavior
    
        Returns
        -------
        None
        
        Notes
        -----
        Adds the Material to the FEM structure

        """
        
        self.material = material
        
    def SetupGMGInterpolaters(self, Nf, maxCoarse=100, maxLevels=5):
        """ Creates restriction matrices for geometric multigrid operations
    
        Parameters
        ----------
        Nf : list of integer
            Number of grid points in the x and y direction in the fine grid
        maxCoarse : integer
            Maximum nodes on the coarse level
        maxLevels : integer
            Maximum number of MG levels
    
        Returns
        -------
        None

        """
        
        temp = np.ones(self.nDof, dtype=int)
        temp[:len(Nf)] = Nf
        Nf = temp
        
        self.P = []
        self.cNodes = []
        nodes = np.arange(np.prod(Nf)).reshape(Nf[::-1])
        for i in range(maxLevels-1):
            Nc = (Nf+1) // 2
            Pi = []
            # Construct P
            for dim in range(self.nDof):
                if Nf[dim] <= 2:
                    Pi.append(np.ones((Nf[dim], 1)))
                else:
                    inds = np.zeros((3*Nc[dim], 2), dtype=int)
                    vals = np.zeros(3*Nc[dim], dtype=float)
                    inds[:2] = [[0, 0], [1, 0]]
                    vals[:2] = [1, 0.5]
                for j in range(1, Nc[dim]-1):
                    inds[3*j-1:3*j+2,:] = np.array([[2*j-1, 2*j, 2*j+1], [j, j, j]]).T
                    vals[3*j-1:3*j+2] = [0.5, 1, 0.5]
                
                if Nf[dim] % 2 == 0:
                    inds[3*Nc[dim]-4:3*Nc[dim]-1,:] = np.array([[2*Nc[dim]-3, 2*Nc[dim]-2, 2*Nc[dim]-1],
                                                                  [Nc[dim]-1, Nc[dim]-1, Nc[dim]]]).T
                    vals[3*Nc[dim]-4:3*Nc[dim]-1] = [0.5, 1, 1]
                    Pi.append(sparse.csr_matrix((vals[:3*Nc[dim]], 
                                         (inds[:3*Nc[dim],0], inds[:3*Nc[dim],1]))))
                    Nc[dim] += 1
                else:
                    inds[3*Nc[dim]-4:3*Nc[dim]-2,:] = np.array([[2*Nc[dim]-3, 2*Nc[dim]-2],
                                                                    [Nc[dim]-1, Nc[dim]-1]]).T
                    vals[3*Nc[dim]-4:3*Nc[dim]-2] = [0.5, 1]
                    Pi.append(sparse.csr_matrix((vals[:3*Nc[dim]-1],
                                     (inds[:3*Nc[dim]-1,0], inds[:3*Nc[dim]-1,1]))))
    
            # Construct P
            P = Pi[0]
            for pi in Pi[1:]:
                P = sparse.kron(pi, P).tocsr()
            self.P.append(sparse.kron(P, np.identity(self.nDof)).tobsr(
                                                blocksize=(self.nDof,self.nDof)))
            
            for i in range(self.nDof):
                if nodes.shape[0] % 2 == 1:
                    nodes = nodes[::2]
                else:
                    nodes = np.concatenate([nodes[::2], nodes[-1:]])
                nodes = np.moveaxis(nodes, 0, -1)
            self.cNodes.append(nodes.ravel())
            
            Nf = Nc
            if np.prod(Nf) <= maxCoarse:
                break

    def Plot(self):
        """ Plots the mesh structure
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None

        """
        
        if self.nodes.shape[1] == 2:
            collection = PolyCollection(self.nodes[self.elements], edgecolors='k',
                                        facecolors='none')
            
            fig = plt.figure("Mesh", figsize=(12,12), clear=True)
            ax = fig.gca()
            ax.add_collection(collection)
            ax.axis('equal')
            ax.axis('off')
            
            # Add loads
            loadNodes = np.where(self.F)[0] // self.nDof
            ax.scatter(self.nodes[loadNodes,0], self.nodes[loadNodes,1], c='r',
                       marker='^', s=100)
            
            # Add supports
            suppNodes = self.fixDof // self.nDof
            ax.scatter(self.nodes[suppNodes,0], self.nodes[suppNodes,1], c='b',
                       marker='s', s=100)
            
            # Add springs
            springNodes = self.springDof // self.nDof
            ax.scatter(self.nodes[springNodes,0], self.nodes[springNodes,1], c='g',
                       marker=r'$\xi$', s=100)
            
        elif self.nodes.shape[1] == 3 and self.nElem < 5000:
            face = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                             2, 3, 7, 6, 0, 4, 7, 3, 1, 5, 6, 2]).reshape(1,-1,4)
            collection = Poly3DCollection(self.nodes[self.elements[:,face].reshape(-1,4)],
                                         facecolors="k", edgecolors="k", alpha=0)
            
            fig = plt.figure("Mesh", figsize=(12,12), clear=True)
            ax = fig.gca(projection='3d')
            ax.add_collection3d(collection)
            ax.set_xlim(np.min(self.nodes), np.max(self.nodes))
            ax.set_ylim(np.min(self.nodes), np.max(self.nodes))
            ax.set_zlim(np.min(self.nodes), np.max(self.nodes))
            
            # Add loads
            loadNodes = np.where(self.F)[0] // self.nDof
            ax.scatter(self.nodes[loadNodes,0], self.nodes[loadNodes,1],
                       self.nodes[loadNodes,2], c='r', marker='^', s=100)
            
            # Add supports
            suppNodes = self.fixDof // self.nDof
            ax.scatter(self.nodes[suppNodes,0], self.nodes[suppNodes,1],
                       self.nodes[suppNodes,2], c='b', marker='s', s=100)
            
            # Add springs
            springNodes = self.springDof // self.nDof
            ax.scatter(self.nodes[springNodes,0], self.nodes[springNodes,1],
                       self.nodes[springNodes,2], c='g', marker=r'$\xi$', s=100)
    
    # Finite element methods
    def Initialize(self):
        """ Sets up auxiliary data structures for assembling the linear system
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        
        self.i = np.zeros(sum([(el.size*self.nDof)**2 for el in self.elements]), dtype=int)
        self.j = self.i.copy()
        self.e = self.i.copy()
        self.k = np.zeros(self.i.shape)
        self.DB = []
        self.G = []
        
        if self.uniform:
            Ke, DB, G = self.LocalK(self.nodes[self.elements[0]])
            self.DB.append(DB)
            self.G.append(G)
            
        ind = 0
        for el in range(self.nElem):
            if not self.uniform:
                Ke, DB, G = self.LocalK(self.nodes[self.elements[0]])[0]
                self.DB.append(DB)
                self.G.append(G)
                
            dof = (self.nDof*self.elements[el].reshape(-1,1) +
                   np.arange(self.nDof).reshape(1,-1)).ravel()
            I, J = np.meshgrid(dof, dof)
            self.i[ind:ind+Ke.size] = I.ravel()
            self.j[ind:ind+Ke.size] = J.ravel()
            self.k[ind:ind+Ke.size] = Ke.ravel()
            self.e[ind:ind+Ke.size] = el
            ind += Ke.size
        
    def ConstructSystem(self, E):
        """ Constructs the linear system by defining K and F.  Does not solve
        the system (must call SolveSystem()).
    
        Parameters
        ----------
        E : array_like
            The densities of each element
    
        Returns
        -------
        None

        """
        # Initial construction
        self.K = sparse.bsr_matrix((self.k*E[self.e], (self.i, self.j)),
                                   blocksize=(self.nDof, self.nDof))
        
        # Add any springs
        springK = np.zeros(self.U.size)
        springK[self.springDof] = self.stiff
        self.K += sparse.spdiags(springK, [0], springK.size, springK.size)
        
        # Adjust right-hand-side
        self.b = self.F - self.K.tocsr()[:, self.fixDof] * self.U[self.fixDof]
        self.b[self.fixDof] = self.U[self.fixDof]
        
        # Apply Dirichlet BC
        interiorDiag = np.zeros(self.K.shape[0])
        interiorDiag[self.freeDof] = 1.
        interiorDiag = sparse.spdiags(interiorDiag, 0, self.K.shape[0],
                                      self.K.shape[1]).tobsr(blocksize=(self.nDof, self.nDof))
        exteriorDiag = np.zeros(self.K.shape[0])
        exteriorDiag[self.fixDof] = 1.
        exteriorDiag = sparse.spdiags(exteriorDiag, 0, self.K.shape[0],
                                      self.K.shape[1]).tobsr(blocksize=(self.nDof, self.nDof))
        self.K = interiorDiag * self.K * interiorDiag + exteriorDiag
        self.K = self.K.tobsr(blocksize=(self.nDof, self.nDof))
        
    def SolveSystemDirect(self, method=None, x0=None):
        """ Solves the linear system for displacements directly
    
        Parameters
        ----------
        method : scipy.sparse.linalg solver
            Ignored
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        
#        self.U[self.freeDof] = spla.spsolve(self.K[self.freeDof[:,np.newaxis],
#                               self.freeDof], self.F[self.freeDof] -
#                               self.K[self.freeDof[:,np.newaxis], self.fixDof] *
#                               self.U[self.fixDof])
        
        diag = self.K.diagonal()
        attachDof = np.where(diag!=0)[0]
        freeDof = np.intersect1d(attachDof, self.freeDof)
        
        K = self.K.tocsr()
        K = K[freeDof, :]
        K = K[:, freeDof]
        K = K.tocoo()
        
        K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
        self.Kfact = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K, self.Kfact)
        B = cvxopt.matrix(self.F[freeDof])
        
        cvxopt.cholmod.solve(self.Kfact,B)
        self.U[:] = 0
        self.U[freeDof] = np.array(B).ravel()
        
        return 1
        
    def SetupAMG(self, maxLevels=None, maxCoarse=None,
                       smoother=('block_jacobi', {'omega':0.5, 'withrho':False})):
        """ Sets up the algebraic multigrid preconditioner
    
        Parameters
        ----------
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        if maxLevels is None:
            maxLevels = self.maxLevels
        if maxCoarse is None:
            maxCoarse = self.maxCoarse
        
        if self.nDof == 1:
            Nullspace = np.ones((self.nodes.shape[0], 1))
        elif self.nDof == 2:
            Nullspace = np.zeros((self.nodes.size,3))
            Nullspace[::2,0]  = 1
            Nullspace[1::2,1] = 1
            Nullspace[::2,2]  = -self.nodes[:,1]
            Nullspace[1::2,2] =  self.nodes[:,0]
            Nullspace = np.linalg.solve(np.linalg.cholesky(np.dot(Nullspace.T,Nullspace)),Nullspace.T).T
        elif self.nDof == 3:
            Nullspace = np.zeros((self.nodes.size,6))
            Nullspace[::3,0]  = 1
            Nullspace[1::3,1] = 1
            Nullspace[2::3,2] = 1
            Nullspace[::3,3]  = -self.nodes[:,1]
            Nullspace[1::3,3] =  self.nodes[:,0]
            Nullspace[1::3,4]  = -self.nodes[:,2]
            Nullspace[2::3,4] =  self.nodes[:,1]
            Nullspace[0::3,5]  = -self.nodes[:,2]
            Nullspace[2::3,5] =  self.nodes[:,0]
            Nullspace = np.linalg.solve(np.linalg.cholesky(np.dot(Nullspace.T,Nullspace)),Nullspace.T).T
        
        self.ml_AMG = pyamg.smoothed_aggregation_solver(self.K,
                                               B=Nullspace, max_coarse=maxCoarse, max_levels=maxLevels,
                                               presmoother=smoother, postsmoother=smoother,
                                               strength=('symmetric',{'theta':0.003}),
                                               coarse_solver='splu', smooth=('jacobi',
                                               {'omega': 4.0/3.0}), keep=True)
        
    def SolveSystemAMG(self, method=None, x0=None, maxLevels=None, maxCoarse=None,
                       smoother=('block_jacobi', {'omega':0.5, 'withrho':False})):
        """ Solves the linear system for displacements using algebraic multigrid
    
        Parameters
        ----------
        method : scipy.sparse.linalg solver
            Optional iterative method to pair with the GMG preconditioner
        x0 : array_like, optional
            Initial guess
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """

        self.SetupAMG(maxLevels, maxCoarse, smoother)
        
        counter = it_counter()
        if method is None:        
            self.U, info = self.ml_AMG.solve(self.b, x0=x0, maxiter=0.1*self.K.shape[0],
                                         tol=1e-8, callback=counter)
        else:
            M = self.ml_AMG.aspreconditioner()
            self.U, info = method(self.ml_AMG.levels[0].A, self.b, x0=x0, tol=1e-8, M=M,
                                  maxiter=0.03*self.K.shape[0], callback=counter)
        self.U[self.fixDof] = 0.
        
        return counter.it
    
    def SetupHybrid(self, maxLevels=None, maxCoarse=None,
                          smoother=('block_jacobi', {'omega':0.5, 'withrho':False}),
                          nG=1):
        """ Sets up the hybrid multigrid
    
        Parameters
        ----------
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
        nG : int, optional
            Number of levels of geometric coarsening to use
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        if maxLevels is None:
            maxLevels = self.maxLevels
        if maxCoarse is None:
            maxCoarse = self.maxCoarse
        
        if self.nDof == 1:
            Nullspace = np.ones((self.nodes.shape[0], 1))
        elif self.nDof == 2:
            Nullspace = np.zeros((self.nodes.size,3))
            Nullspace[::2,0]  = 1
            Nullspace[1::2,1] = 1
            Nullspace[::2,2]  = -self.nodes[:,1]
            Nullspace[1::2,2] =  self.nodes[:,0]
        elif self.nDof == 3:
            Nullspace = np.zeros((self.nodes.size,6))
            Nullspace[::3,0]  = 1
            Nullspace[1::3,1] = 1
            Nullspace[2::3,2] = 1
            Nullspace[::3,3]  = -self.nodes[:,1]
            Nullspace[1::3,3] =  self.nodes[:,0]
            Nullspace[1::3,4]  = -self.nodes[:,2]
            Nullspace[2::3,4] =  self.nodes[:,1]
            Nullspace[0::3,5]  = -self.nodes[:,2]
            Nullspace[2::3,5] =  self.nodes[:,0]
            
        levels = []
        A = self.K
        for i in range(nG):
            levels.append(pyamg.multilevel_solver.level())
            levels[-1].A = A
            levels[-1].P = self.P[i]
            levels[-1].R = self.P[i].T
            A = levels[-1].R * A * levels[-1].P
            
        if nG > 0:
            cDof = (self.nDof*self.cNodes[nG-1].reshape(-1,1) + np.arange(self.nDof).reshape(1,-1)).ravel()
            Nullspace = Nullspace[cDof]
        Nullspace = np.linalg.solve(np.linalg.cholesky(np.dot(Nullspace.T,Nullspace)),Nullspace.T).T
    
        AMG = pyamg.smoothed_aggregation_solver(A,
                                                B=Nullspace, max_coarse=maxCoarse,
                                                max_levels=maxLevels-nG+1,
                                                strength=('symmetric',{'theta':0.003}),
                                                smooth=('jacobi', {'omega': 4.0/3.0}),
                                                keep=True)
        

        levels += AMG.levels
        self.ml_HYBRID = pyamg.multilevel_solver(levels, coarse_solver='splu')
        from pyamg.relaxation.smoothing import change_smoothers
        change_smoothers(self.ml_HYBRID, presmoother=smoother, postsmoother=smoother)
        
    def SolveSystemHybrid(self, method=None, x0=None, maxLevels=None, maxCoarse=None,
                          smoother=('block_jacobi', {'omega':0.5, 'withrho':False}),
                          nG=1):
        """ Solves the linear system for displacements using hybrid multigrid
    
        Parameters
        ----------
        method : scipy.sparse.linalg solver
            Optional iterative method to pair with the GMG preconditioner
        x0 : array_like, optional
            Initial guess
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
        nG : int, optional
            Number of levels of geometric coarsening to use
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        
        self.SetupHybrid(maxLevels, maxCoarse, smoother, nG)
        
        counter = it_counter()
        if method is None:        
            self.U, info = self.ml_HYBRID.solve(self.b, x0=x0, maxiter=0.1*self.K.shape[0],
                                         tol=1e-8, callback=counter)
        else:
            M = self.ml_HYBRID.aspreconditioner()
            self.U, info = method(self.K, self.b, x0=x0, tol=1e-8, M=M,
                                  maxiter=5000, callback=counter)
        self.U[self.fixDof] = 0.
        
        return counter.it
        
    def SetupGMG(self, maxLevels=None, maxCoarse=None,
                       smoother=('block_jacobi', {'omega' : 0.5, 'withrho': False})):
        """ Sets up the geometric multigrid
    
        Parameters
        ----------
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        
        if maxLevels is None:
            maxLevels = self.maxLevels
        if maxCoarse is None:
            maxCoarse = self.maxCoarse
                
        levels = []
        levels.append(pyamg.multilevel_solver.level())
        levels[-1].A = self.K
        for P in self.P:
            levels[-1].P = P
            levels[-1].R = P.T
            levels.append(pyamg.multilevel_solver.level())
            levels[-1].A = levels[-2].R * levels[-2].A * levels[-2].P
            if (len(levels) == maxLevels or
                levels[-1].A.shape[0]//levels[1].A.blocksize[0] < maxCoarse):
                break
            
        self.ml_GMG = pyamg.multilevel_solver(levels, coarse_solver='splu')
        from pyamg.relaxation.smoothing import change_smoothers
        change_smoothers(self.ml_GMG, presmoother=smoother, postsmoother=smoother)
        
    def SolveSystemGMG(self, method=None, x0=None, maxLevels=None, maxCoarse=None,
                       smoother=('block_jacobi', {'omega' : 0.5, 'withrho': False})):
        """ Solves the linear system for displacements using geometric multigrid
    
        Parameters
        ----------
        method : scipy.sparse.linalg solver
            Optional iterative method to pair with the GMG preconditioner
        x0 : array_like, optional
            Initial guess
        maxLevels : int, optional
            Maximum levels in the hierarchy
        maxCoarse : int, optional
            Maximum nodes on the coarse grid
        smoother : tuple or list of tuple, optional
            Describes the smoothers to use on each level
    
        Returns
        -------
        it : integer
            Number of solver iterations

        """
        
        self.SetupGMG(maxLevels, maxCoarse, smoother)
        
        counter = it_counter()
        if method is None:        
            self.U, info = self.ml_GMG.solve(self.b, x0=x0, maxiter=0.1*self.K.shape[0],
                                         tol=1e-8, callback=counter)
        else:
            M = self.ml_GMG.aspreconditioner()
            self.U, info = method(self.ml_GMG.levels[0].A, self.b, x0=x0, tol=1e-8, M=M,
                                  maxiter=0.03*self.K.shape[0], callback=counter)
        
        self.U[self.fixDof] = 0.
        
        return counter.it
    
    def LocalK(self, coordinates):
        """ Constructs a local stiffness matrix
        
        Parameters
        ----------
        coordinates : array_like
            Coordinates of the element being assembled
        
        Returns
        -------
        Ke : array_like
            Local Stiffness matrix
        DB : array_like
            Constitutive matrix times shape function matrix for
            converting displacements to stresses
        G : array_like
            Shape function matrix for constructing stress stiffness matrix
        
        """
        
        GP, w = Shape_Functions.QuadRule(coordinates.shape[0], coordinates.shape[1])
        Ke = np.zeros((coordinates.shape[0]*self.nDof,coordinates.shape[0]*self.nDof))
        DB = []
        G = []
        for q in range(w.size):
            dNdxi = Shape_Functions.dN(GP[q], coordinates.shape[0])[0]
            J = np.dot(dNdxi.T, coordinates)
            detJ = np.linalg.det(J)
            dNdx = np.linalg.solve(J, dNdxi.T)
            B = Shape_Functions.AssembleB(dNdx)
            DB.append(w[q] * detJ * np.dot(self.material.Get_D(), B))
            G.append(Shape_Functions.AssembleG(dNdx))
            Ke += np.dot(B.T, DB[-1]) * w[q]
            
        return Ke, np.concatenate(DB), np.concatenate(G)
        