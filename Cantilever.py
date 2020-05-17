# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:36:52 2019

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from FEM import FEM
import Material
import Interpolation
import Update
from Optimization import PyOpt

#def run():
fem = FEM()
Nelx = 100
Nely = 50
Dimensions = [0, 10, 0, 5]
fem.Create2DMesh(Dimensions, Nelx, Nely, maxLevels=10)
fem.SolveSystem = fem.SolveSystemDirect
ofst = 1e-10

lower = np.array([10-ofst, 2.5-ofst])
upper = np.array([10+ofst, 2.5+ofst])
loads = [0, -1e4]
loadSpecs = [{'lower':lower, 'upper':upper, 'force':loads}]
fem.AddLoad(loadSpecs)

lower = np.array([0-ofst, 0-ofst])
upper = np.array([0+ofst, 5+ofst])
bc = [0, 0]
bcSpecs = [{'lower':lower, 'upper':upper, 'disp':bc}]
fem.AddBC(bcSpecs)

#    springpoly = np.array([[-1, 0.5-ofst], [2, 0.5-ofst], [2, 0.5+ofst], [-1, 0.5+ofst]])
#    stiff = [1, 1]
#    springSpecs = [{'poly':springpoly, 'stiff':stiff}]
#    fem.AddSprings(springSpecs)

#fem.Plot()

fem.SetMaterial(Material.PlaneStressElastic(15e9, 0.3))

#update = Update.OCUpdateScheme(0.2, 0.5, np.linspace(0, 1, fem.nElem, endpoint=False),#0.5 * np.ones(fem.nElem),
#                               np.zeros(fem.nElem), np.ones(fem.nElem))
update = Update.MMA(0.5 * np.ones(fem.nElem), 1, np.zeros(fem.nElem), np.ones(fem.nElem))

opt = PyOpt(fem, update)

# Minimum feature size filter
radius = 1.5 * (Dimensions[1] - Dimensions[0]) / Nelx
opt.Filter = opt.ConstructDensityFilter(radius=radius, nElx=[Nelx, Nely])

# Maximum feature size filter
radius = 0.5 * (Dimensions[1] - Dimensions[0]) / Nelx
opt.R = opt.ConstructDensityFilter(radius=radius, nElx=[Nelx, Nely])
rowsum = opt.R.indptr[1:] - opt.R.indptr[:-1]
edge = rowsum.max() - rowsum
opt.R = opt.R.tocoo()
opt.R = sparse.csr_matrix((np.concatenate([1+0*opt.R.data, edge]),
                           (np.concatenate([opt.R.row, np.arange(opt.R.shape[0])]),
                            np.concatenate([opt.R.col, opt.R.shape[0]*np.ones(opt.R.shape[0], dtype=int)]))))

fem.Initialize()

from Functions import Compliance, Volume, Stability
opt.AddFunction(Compliance, 1., 0, 1, 'objective')
opt.AddFunction(Volume, 0.4, 0, 1, 'constraint')


for penal in np.linspace(1,4,7):
    print("Penalty set to %1.2f" % penal)
    opt.SetInterpolation(Interpolation.SIMP_CUT(opt.Filter, opt.R, 50, penal,
                                                0, 0.01, minStiff=1e-10))
    opt.Optimize(maxit=30, plt_freq=10)