# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:30:43 2019

@author: Darin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.sparse as sparse
import Material
import Update

class PyOpt:
    """ Topology optimization object
    """
    
    def __init__(self, fem=None, update=None, threshold=0.3):
        """Constructor 
        
        Parameters
        ----------
        fem : FEM object
            An object describing the underlying finite element analysis
        update : Update scheme object
            Provides functionality to store and update design variables
        threshold : scalar
            Minimum density to plot 3D elements
        """
        
        self.fem = fem
        self.update = update
        self.dens_thresh = threshold
        self.objectives = []
        self.constraints = []
        self.f = []
        self.g = []
        
    def LoadFromFile(self, filename):
        """ Loads an old run from a file
        
        Parameters
        ----------
        filename : string
            Name of the file that everything will be saved to
        
        Returns
        -------
        None
            
        """
        
        data = np.load(filename, allow_pickle=True).item()
        from FEM import FEM
        self.fem = FEM()
        self.fem.Load(data['fem'])
        
        if data['update']['type'] == 'OC':
            self.update = Update.OCUpdateScheme(0.2, 0.5, 0.5 * np.ones(self.fem.nElem),
                                      np.zeros(self.fem.nElem), np.ones(self.fem.nElem))
        elif data['update']['type'] == 'MMA':
            self.update = Update.MMA(0.5 * np.ones(self.fem.nElem), 1,
                        np.zeros(self.fem.nElem), np.ones(self.fem.nElem))
        self.update.Load(data['update'])
                
        self.Filter = data['opt']['Filter']
        try:
            self.R = data['opt']['R']
        except:
            pass
        import Functions as Funcs
        for objective in data['opt']['objectives']:
            self.AddFunction(getattr(Funcs, objective['function']),
                             objective['weight'], objective['min'],
                             objective['max'], 'objective')
        for constraint in data['opt']['constraints']:
            self.AddFunction(getattr(Funcs, constraint['function']),
                             constraint['constraint'], constraint['min'],
                             constraint['max'], 'constraint')
            
    def LoadPetsc(self, folder, appendix=None, Endian='=', update='MMA'):
        """ Create PyOpt structure from PETSc code results
        
        Parameters
        ----------
        folder : str
            folder containing all of the Petsc results
        appendix : str
            Appendix for result values to restart from, if none picks highest penalty
        Endian : char
            Indicates byte ordering ('=':default, '<':little Endian, '>':big Endian)
        update : str
            Which updte scheme to use (MMA or OC)
        
        Returns
        -------
        None
            
        """
        
        from os import listdir
        from os.path import sep
        from PetscBinaryIO import PetscBinaryRead
        import Functions_Timing as Funcs
        
        # Load FEM data
        from FEM import FEM
        self.fem = FEM()
        self.fem.LoadPetsc(folder, Endian=Endian)
        
        # Load update data
        if update == 'OC':
            self.update = Update.OCUpdateScheme(0.2, 0.5, 0.5 * np.ones(self.fem.nElem),
                                      np.zeros(self.fem.nElem), np.ones(self.fem.nElem))
        elif update == 'MMA':
            self.update = Update.MMA(0.5 * np.ones(self.fem.nElem), 1,
                        np.zeros(self.fem.nElem), np.ones(self.fem.nElem))
        self.update.LoadPetsc(folder, appendix=appendix, Endian=Endian)
              
        # Load filter matrics
        self.Filter = PetscBinaryRead(folder + sep + "Filter.bin")
        try:
            self.R = PetscBinaryRead(folder + sep + "Max_Filter.bin")
            edge = PetscBinaryRead(folder + sep + "Void_Edge_Volume.bin")
            self.R = self.R.tocoo()
            self.R = sparse.csr_matrix((np.concatenate([self.R.data, edge]),
                                       (np.concatenate([self.R.row, np.arange(self.R.shape[0])]),
                                        np.concatenate([self.R.col, self.R.shape[0]*np.ones(self.R.shape[0], dtype=int)]))))
        except:
            self.R = sparse.dia_matrix((np.ones(self.fem.nElem), np.zeros(self.fem.nElem)))
        
        # Set up functions and material properties
        inputFile = [file for file in listdir(folder) if '_Input' in file][0]
        active = False
        name = None
        fType = None
        value = None
        minimum = None
        maximum = None
        E0, Nu0, Density = None, None, None
        with open(folder + sep + inputFile, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line[:3] == 'E0:':
                    E0 = float(line.split(':')[-1])
                elif line[:4] == 'Nu0:':
                    Nu0 = float(line.split(':')[-1])
                elif line[:8] == 'Density:':
                    Density = float(line.split(':')[-1])
                elif '[Functions]' in line:
                    active = True
                elif '[/Functions]' in line:
                    active = False
                elif active:
                    if line in ['Compliance', 'Stability', 'Frequencey', 'Volume']:
                        name = line
                    elif line in ['Objective', 'Constraint']:
                        fType = line
                    elif 'Values:' in line:
                        value = [float(val) for val in line.split(':')[-1].split(',')][0]
                    elif 'Range:' in line:
                        minimum, maximum = [float(val) for val in line.split(':')[-1].split(',')]
                    if name is not None and fType is not None and value is not None and minimum is not None:
                        self.AddFunction(getattr(Funcs, name), value, minimum, maximum, fType)
                        name = None
                        fType = None
                        value = None
                        minimum = None
                        maximum = None
          
        if self.fem.nDof == 2:             
            self.fem.SetMaterial(Material.PlaneStressElastic(E0, Nu0))
        else:
            self.fem.SetMaterial(Material.Elastic3D(E0, Nu0))
        
        
    def SetInterpolation(self, interpolation):
        """ Set the object for interpolating filtered densities to material values
        
        Parameters
        ----------
        interpolation : Interpolation object
            The interpolation object
        
        Returns
        -------
        None
            
        """
        self.MatIntFnc = interpolation.Interpolate
        
    def ConstructDensityFilter(self, radius, nElx):
        """ Sets up the density filter
        
        Parameters
        ----------
        radius : scalar
            Filter radius
        nElx : list of integer
            Number of elements in each direction
        
        Returns
        -------
        Filter : sparse matrix
            Filter matrix
            
        """
                
        centroids = np.mean(self.fem.nodes[self.fem.elements.reshape(
                        1,self.fem.nElem,-1)], axis=2).reshape(self.fem.nElem,-1)
    
        # Element sizes
        dx = np.zeros(self.fem.nodes.shape[1])
        # Number of elements to check in each direction
        Nx = np.zeros(self.fem.nodes.shape[1], dtype=int)
        for i in range(dx.size):
            dx[i] = (np.max(self.fem.nodes[self.fem.elements[0], i]) -
                     np.min(self.fem.nodes[self.fem.elements[0], i]))
            Nx[i] = max(np.floor(radius/dx[i]), 1)
        
        # Distance of all nearby elements
        offset = [np.arange(-Nx[0], Nx[0]+1)]
        for i in range(1, self.fem.nodes.shape[1]):
            newshape = [1 for j in range(i)] + [2*Nx[i]+1]
            for j in range(len(offset)):
                offset[j] = np.tile(np.expand_dims(offset[j], axis=-1), newshape)
            newshape = [1 for j in range(i)] + [-1]
            offset.append(np.arange(-Nx[i], Nx[i]+1).reshape(newshape))
            newshape = list(offset[0].shape)
            newshape[-1] = 1
            offset[-1] = np.tile(offset[-1], newshape)

        dist = [dx[i]*d.ravel() for i, d in enumerate(offset)]
        r = np.sqrt(np.array([d**2 for d in dist]).sum(axis=0))
        Nbrhd = r < radius        
        Low_Bnd = np.min(self.fem.nodes, axis=0)
        Upp_Bnd = np.max(self.fem.nodes, axis=0)
        sx = [1]
        for nEl in nElx[:-1]:
            sx.append(sx[-1] * nEl)
        Template = sum([sx[i]*d for i, d in enumerate(offset)]).ravel()
        
        indi = [0 for i in range(self.fem.nElem)]
        indj = [0 for i in range(self.fem.nElem)]
        valk = [0 for i in range(self.fem.nElem)]
        for el in range(self.fem.nElem):
            Add = el + Template
            Valid = [np.logical_and(centroids[el, i]+dist[i] > Low_Bnd[i],
                                    centroids[el, i]+dist[i] < Upp_Bnd[i])
                     for i in range(len(dist))]
            Valid = np.logical_and.reduce(Valid)
            Valid = np.logical_and(Valid, Nbrhd)
            Add = Add[Valid]
            
            indi[el] = Add
            indj[el] = el*np.ones(len(Add), dtype=int)
            valk[el] = r[Valid]

        Filter = sparse.csr_matrix((1-np.concatenate(valk)/radius,
                                        (np.concatenate(indi),np.concatenate(indj))))
        rowsum = Filter.sum(axis=1)
        return sparse.dia_matrix((1/rowsum.T,0),shape=Filter.shape) * Filter    
        
    def AddFunction(self, function, value, minimum, maximum, funcType):
        """ Add an objective or constraint function to the list of functions
        to be evaluated
        
        Parameters
        ----------
        function : OptFunction object
            The objective function. Returns a function value and design sensitivities
        value : scalar
            The objective weight or constraint value.
            Objective weights should be adjusted so all weights sum to 1.
        minimum : scalar
            Mimimum function value for normalization
        minimum : scalar
            Mimimum function value for normalization
        funcType : str
            'objective' or 'constraint'
        
        Returns
        -------
        None
            
        """
        
        if funcType.lower() == 'objective':
            self.objectives.append({'function':function, 'weight':value,
                                    'min':minimum, 'max':maximum})
        else:
            self.constraints.append({'function':function, 'constraint':value,
                                     'min':minimum, 'max':maximum})
            
    def CallFunctions(self):
        """ Call all functions to get objective and constraint value as well
        as all function sensitivities
        
        Parameters
        ----------
        None
        
        Returns
        -------
        f : scalar
            Objective value
        dfdx : array_like
            Objective gradients
        g : array_like
            Constraint values
        dgdx : array_like
            Constraint gradients
            
        """
        
        matVals = self.MatIntFnc(self.update.x)
        self.densities = matVals['V']
        self.fem.ConstructSystem(matVals['E'])
        
        x0 = self.fem.U.copy()
        self.fem.SolveSystem(sparse.linalg.cg, x0=x0)

        f = 0
        dfdx = np.zeros(self.fem.nElem)
        g = np.zeros(max(self.update.m, 1))
        dgdx = np.zeros((self.fem.nElem, g.size))
        
        for funDict in self.objectives:
            obj, dobjdE, dobjdV = funDict['function'](self.fem, matVals)
            dobjdx = self.Filter.T * dobjdV
            dobjdx += self.Filter.T * (matVals['y'] * dobjdE -
                                       matVals['rhoq'] * (matVals['y'] < 1) * 
                                       (self.R.T[:-1,:] * (matVals['rho'] * dobjdE)))
            f += funDict['weight'] * (obj - funDict['min']) / (funDict['max'] -
                                                               funDict['min'])
            dfdx += funDict['weight'] * dobjdx / (funDict['max'] - funDict['min'])
            print("\t%s: %f" % (funDict['function'].__name__, 
                                  funDict['weight'] * (obj - funDict['min']) /
                                  (funDict['max'] - funDict['min'])))
            
        i = 0
        for iiii, funDict in enumerate(self.constraints):
            con, dcondE, dcondV = funDict['function'](self.fem, matVals)
            dcondx = self.Filter.T * dcondV
            dcondx += self.Filter.T * (matVals['y'] * dcondE -
                                       matVals['rhoq'] * (matVals['y'] < 1) * 
                                       (self.R.T[:-1,:] * (matVals['rho'] * dcondE)))
            g[i] = (con - funDict['constraint']) / (funDict['max'] - funDict['min'])
            dgdx[:,i] = dcondx / (funDict['max'] - funDict['min'])
            i += 1
            print("\t%s: %f" % (funDict['function'].__name__, g[i-1]))
        
        self.f.append(f)
        self.g.append(g)
        return f, dfdx, g, dgdx
            
    def Plot(self, filename=None, edgeColor='none'):
        """ Plot the optimized shape
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            
        """
        
        fig = plt.figure("Result", figsize=(12,12), clear=True)
        if self.fem.nDof == 2:
            collection = PolyCollection(self.fem.nodes[self.fem.elements], edgecolors=edgeColor)
            collection.set_array(self.densities)
            collection.set_cmap('gray_r')
            collection.set_clim(vmin=0, vmax=1)
            ax = fig.gca()
            ax.add_collection(collection)
            ax.set_xlim(self.fem.nodes[:,0].min(), self.fem.nodes[:,0].max())
            ax.set_ylim(self.fem.nodes[:,1].min(), self.fem.nodes[:,1].max())
            ratio = ((ax.get_ylim()[1] - ax.get_ylim()[0]) /
                     (ax.get_xlim()[1] - ax.get_xlim()[0]))
            if ratio < 1:
                fig.set_figheight(ratio * fig.get_figwidth())
            else:
                fig.set_figwidth(fig.get_figheight() / ratio)
            ax.axis('off')
        elif self.fem.nDof == 3:
            if not hasattr(self, 'facePairs'):
                face = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                                 3, 2, 6, 7, 0, 4, 7, 3, 1, 5, 6, 2]).reshape(6,4)
                faces = np.concatenate([el[face] for el in self.fem.elements])
                order = np.arange(faces.shape[0])
                for i in range(faces.shape[1]):
                    ind = np.argsort(faces[:,i], kind='stable')
                    faces = faces[ind,:]
                    order = order[ind]
                elements = np.tile(np.arange(self.fem.nElem).reshape(-1, 1), (1,6)).ravel()[order]
                self.facePairs = []
                self.faceNodes = []
                i = 0
                while i < faces.shape[0] - 1:
                    if (faces[i,:] == faces[i+1,:]).all():
                        self.facePairs.append([elements[i], elements[i+1]])
                        self.faceNodes.append(faces[i])
                        i += 2
                    else:
                        self.facePairs.append([elements[i], self.fem.nElem])
                        self.faceNodes.append(faces[i])
                        i += 1
                if i < faces.shape[0]:
                    self.facePairs.append([elements[i], self.fem.nElem])
                    self.faceNodes.append(faces[i])
                        
                self.facePairs = np.array(self.facePairs)
                self.faceNodes = np.array(self.faceNodes)
                
            densities = np.append(self.densities, [0])
            faces = np.logical_xor(densities[self.facePairs[:,0]] > self.dens_thresh,
                                   densities[self.facePairs[:,1]] > self.dens_thresh)
            print("Plotting %i faces" % (self.faceNodes[faces].size // 4))
            collection = Poly3DCollection(self.fem.nodes[self.faceNodes[faces].reshape(-1,4)],
                                          edgecolors=edgeColor)
            densities = densities[self.facePairs].max(axis=1)[faces]
            collection.set_array(densities)
                
#            elements = self.fem.elements[self.densities > self.dens_thresh,:]
#            collection = Poly3DCollection(self.fem.nodes[elements[:,face].reshape(-1,4)],
#                                          edgecolors="k")
#            densities = np.tile(self.densities[self.densities > self.dens_thresh], (6,1)).T.ravel()
#            collection.set_array(densities)
            collection.set_cmap('gray_r')
            collection.set_clim(vmin=0, vmax=1)
            
            ax = fig.gca(projection='3d')
            ax.add_collection3d(collection)
            ax.set_xlim(np.min(self.fem.nodes), np.max(self.fem.nodes))
            ax.set_ylim(np.min(self.fem.nodes), np.max(self.fem.nodes))
            ax.set_zlim(np.min(self.fem.nodes), np.max(self.fem.nodes))
            
        plt.draw()
        plt.pause(0.01)
        if filename:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        
    def Optimize(self, maxit=100, tol=1e-5, plt_freq=1, filename=None):
        """ Iteratively improve the structure
        
        Parameters
        ----------
        maxit : integer
            Maximum number of iterations to run
        tol : scalar
            Minimum change in a single design variable to continue iterating
        plt_freq : int
            Number of iterations between plotting
        
        Returns
        -------
        None
            
        """
        
        for it in range(maxit):
            f, dfdx, g, dgdx = self.CallFunctions()
            print(it, f, g)
            change = self.update.Update(dfdx, g, dgdx)
            if (it+1) % plt_freq == 0:
                if filename:
                    self.Plot(filename + '_it%i.png' % it)
                else:
                    self.Plot()
            
            if change < tol:
                break
            
    def Save(self, filename):
        """ Saves the current status of the optimization
        
        Parameters
        ----------
        filename : string
            Name of the file that everything will be saved to
        
        Returns
        -------
        None
            
        """
        
        update_data = self.update.GetData()
        fem_data = self.fem.GetData()
        opt_data = {'Filter':self.Filter, 'R':self.R}
        
        functions = []
        for funDict in self.objectives:
            fun_data = funDict.copy()
            fun_data['function'] = funDict['function'].__name__
            functions.append(fun_data)
        opt_data['objectives'] = functions
        
        functions = []
        for funDict in self.constraints:
            fun_data = funDict.copy()
            fun_data['function'] = funDict['function'].__name__
            functions.append(fun_data)
        opt_data['constraints'] = functions
        
        np.save(filename, {'update':update_data, 'fem':fem_data, 'opt':opt_data})