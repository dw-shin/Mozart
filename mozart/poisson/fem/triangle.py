from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D, Dmatrices2D, VandermondeM1D

def getMatrix(degree):
	"""
	Get FEM matrices on the reference triangle

	Paramters
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference triangle
		- ``Srr_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_r phi_i \partial_r phi_j dr)
		- ``Srs_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_r phi_i \partial_s phi_j dr)
		- ``Ssr_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_s phi_i \partial_r phi_j dr)
		- ``Sss_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_s phi_i \partial_s phi_j dr)
		- ``Dr_R`` (``float64 array``) : Differentiation matrix along r-direction
		- ``Ds_R`` (``float64 array``) : Differentiation matrix along s-direction
		- ``M1D_R`` (``float64 array``) : Mass matrix on the reference interval (1D)

	Example
		>>> N = 1
		>>> M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R, M1D_R = getMatrix(N)
		>>> M_R
		array([[ 0.33333333,  0.16666667,  0.16666667],
		   [ 0.16666667,  0.33333333,  0.16666667],
		   [ 0.16666667,  0.16666667,  0.33333333]])
		>>> Srr_R
		array([[ 0.5, -0.5,  0. ],
		   [-0.5,  0.5,  0. ],
		   [ 0. ,  0. ,  0. ]])
		>>> Srs_R
		array([[  5.00000000e-01,  -9.80781986e-17,  -5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
		>>> Ssr_R
		array([[  5.00000000e-01,  -5.00000000e-01,   0.00000000e+00],
		   [ -9.80781986e-17,   9.80781986e-17,   0.00000000e+00],
		   [ -5.00000000e-01,   5.00000000e-01,   0.00000000e+00]])
		>>> Sss_R
		array([[  5.00000000e-01,  -9.80781986e-17,  -5.00000000e-01],
		   [ -9.80781986e-17,   1.92386661e-32,   9.80781986e-17],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01]])
		>>> Dr_R
		array([[-0.5,  0.5,  0. ],
		   [-0.5,  0.5,  0. ],
		   [-0.5,  0.5,  0. ]])
		>>> Ds_R
		array([[ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01]])
		>>> M1D_R
		array([[ 0.66666667,  0.33333333],
		   [ 0.33333333,  0.66666667]])
	"""

	r, s = RefNodes_Tri(degree)
	V = Vandermonde2D(degree,r,s)
	invV = np.linalg.inv(V)
	M_R = np.dot(np.transpose(invV),invV)
	Dr_R, Ds_R = Dmatrices2D(degree, r, s, V)
	Srr_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Dr_R)
	Srs_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Ds_R)
	Ssr_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Dr_R)
	Sss_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Ds_R)

	r1D = np.linspace(-1, 1, degree+1)
	V1D = VandermondeM1D(degree, r1D)
	invV1D = np.linalg.inv(V1D)
	M1D_R = np.dot(np.transpose(invV1D),invV1D)
	return (M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R, M1D_R)

def getIndex(degree, c4n, n4e, n4sDb, n4sNb):
	"""
	Get indices on each element

	Paramters
		- ``degree`` (``int32``) : degree of polynomial
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4sDb`` (``int32 array``) : nodes for sides on Dirichlet boundary
		- ``n4sNb`` (``int32 array``) : nodes for sides on Neumann boundary

	Returns
		- ``c4nNew`` (``float64 array``) : coordinates for all nodes (vetex, side and interior nodes)
		- ``ind4e`` (``int32 array``) : indices on each element
		- ``ind4Db`` (``int32 array``) : indices on Dirichlet boundary
		- ``ind4Nb`` (``int32 array``) : indices on Neumann boundary

	Example
		>>> N = 3
		>>> c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> n4sDb = np.array([[0, 1], [2, 3], [3, 4]])
		>>> n4sNb = np.array([[1, 2]])
		>>> c4nNew, ind4e, ind4Db, ind4Nb = getIndex(N, c4n, n4e, n4sDb, n4sNb)
		>>> c4nNew
		array([[ 0.        ,  0.        ],
		   [ 1.        ,  0.        ],
		   [ 1.        ,  1.        ],
		   [ 0.        ,  1.        ],
		   [ 0.66666667,  0.33333333],
		   [ 0.33333333,  0.66666667],
		   [ 0.        ,  0.66666667],
		   [ 0.        ,  0.33333333],
		   [ 1.        ,  0.33333333],
		   [ 1.        ,  0.66666667],
		   [ 0.33333333,  0.        ],
		   [ 0.66666667,  0.        ],
		   [ 0.66666667,  1.        ],
		   [ 0.33333333,  1.        ],
		   [ 0.33333333,  0.33333333],
		   [ 0.66666667,  0.66666667]])
		>>> ind4e
		array([[ 0, 10, 11,  1,  7, 14,  4,  6,  5,  3],
		   [ 2, 12, 13,  3,  9, 15,  5,  8,  4,  1]])
		>>> ind4Db
		array([ 0,  1,  2,  3,  4,  4,  5, 10, 11, 12, 13])
		>>> ind4Nb
		array([[1, 8, 9, 2]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	n4sInd = np.sort(ind)
	n4s = allSides[n4sInd,:]

	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(3,-1).transpose().astype('int')

	nrElems = n4e.shape[0]
	elemNumbers = np.hstack((np.hstack((np.arange(0,nrElems),np.arange(0,nrElems))),np.arange(0,nrElems)))

	e4s=np.zeros((ind.size,2),int)
	e4s[:,0]=elemNumbers[n4sInd] + 1

	allElems4s=np.zeros(allSides.shape[0],int)
	tmp2 = np.bincount((back + 1),weights = (elemNumbers + 1))
	allElems4s[ind]=tmp2[1::]
	e4s[:,1] = allElems4s[n4sInd] - e4s[:,0]
	e4s=e4s-1

	sideNr2 = np.zeros(3*nrElems, dtype = int)
	sideNr2[n4sInd] = 1
	S_ind = sideNr2.reshape(n4e.shape[1],n4e.shape[0]).transpose()

	nrNodes = c4n.shape[0]
	nrLocal = int((degree+1)*(degree+2)/2)
	nri = int((degree-2)*(degree-1)/2)
	nNS = nrNodes + (degree-1)*n4s.shape[0]
	BDindex_F = np.array([(np.arange(degree-1,0,-1)*(2*degree+4 - np.arange(degree,1,-1))/2).astype(int),
		np.arange(1,degree), (degree + np.arange(1,degree)*(2*degree+2 - np.arange(2,degree+1))/2).astype(int)])
	Iindex = np.setdiff1d(np.arange(0,nrLocal),np.hstack((BDindex_F.flatten(),np.array([0, degree, nrLocal-1]))))
	ind4e = np.zeros((nrElems,nrLocal), dtype = int)
	ind4e[:,np.array([0, degree, nrLocal-1])] = n4e[:,np.array([2, 0, 1])]
	edge =  (np.tile(S_ind[:,1],(degree-1,1)).transpose()*np.tile(np.arange(0,degree-1),(n4e.shape[0],1))) + \
	   (np.tile((1-S_ind[:,1]),(degree-1,1)).transpose()*np.tile(np.arange(degree-2,-1,-1),(n4e.shape[0],1)))
	ind4e[:,BDindex_F[0]] = nrNodes + np.tile(s4e[:,1]*(degree-1),(degree-1,1)).transpose() + edge
	edge =  (np.tile(S_ind[:,2],(degree-1,1)).transpose()*np.tile(np.arange(0,degree-1),(n4e.shape[0],1))) + \
	   (np.tile((1-S_ind[:,2]),(degree-1,1)).transpose()*np.tile(np.arange(degree-2,-1,-1),(n4e.shape[0],1)))
	ind4e[:,BDindex_F[1]] = nrNodes + np.tile(s4e[:,2]*(degree-1),(degree-1,1)).transpose() + edge
	edge =  (np.tile(S_ind[:,0],(degree-1,1)).transpose()*np.tile(np.arange(0,degree-1),(n4e.shape[0],1))) + \
	   (np.tile((1-S_ind[:,0]),(degree-1,1)).transpose()*np.tile(np.arange(degree-2,-1,-1),(n4e.shape[0],1)))
	ind4e[:,BDindex_F[2]] = nrNodes + np.tile(s4e[:,0]*(degree-1),(degree-1,1)).transpose() + edge
	ind4e[:,Iindex] = np.arange(nNS,nNS+nrElems*nri).reshape(nrElems,nri)

	# Compute boundary information
	indexBDs = (e4s[:,1]==-1).nonzero()[0]
	nrSides = n4s.shape[0]
	from scipy.sparse import coo_matrix
	BDsides_COO = coo_matrix((indexBDs, (n4s[indexBDs,0], n4s[indexBDs,1])), shape=(nrSides, nrSides))
	BDsides_CSR = BDsides_COO.tocsr()
	indexDbs = np.zeros(n4sDb.shape[0], dtype=int)
	for j in range(0,n4sDb.shape[0]):
		indexDbs[j] = BDsides_CSR[n4sDb[j,0], n4sDb[j,1]]

	indexDbs = np.sort(indexDbs)
	nrDbs = indexDbs.shape[0]
	tmp = np.tile(indexDbs * (degree - 1),(degree-1,1)).transpose() + \
	   np.tile(np.arange(0,degree-1),(nrDbs,1))
	ind4Db = np.append(np.unique(n4sDb), nrNodes + tmp.flatten())

	if n4sNb.size != 0:
		ind4Nb = np.zeros((n4sNb.shape[0], degree+1), dtype = int)
		BDindex_F = np.array([(np.arange(degree,-1,-1)*(2*degree+4 - np.arange(degree+1,0,-1))/2).astype(int),
			np.arange(0,degree+1), (degree + np.arange(0,degree+1)*(2*degree+2 - np.arange(1,degree+2))/2).astype(int)])
		for j in range(0,n4sNb.shape[0]):
			sideNb = BDsides_CSR[n4sNb[j,0], n4sNb[j,1]]
			elem = e4s[sideNb, 0]
			ind4Nb[j,:] = ind4e[elem,BDindex_F[s4e[elem,[1,2,0]] == sideNb,:]]

	r, s = RefNodes_Tri(degree)

	if degree > 1:
		r1D = np.linspace(-1,1,degree+1)
		r1D = r1D[1:degree]
		Mid = (c4n[n4s[:,0],:] + c4n[n4s[:,1],:])/2
		c4sx = np.tile(Mid[:,0],(degree-1,1)).transpose() + np.outer((c4n[n4s[:,1],0] - c4n[n4s[:,0],0])/2,r1D)
		c4sy = np.tile(Mid[:,1],(degree-1,1)).transpose() + np.outer((c4n[n4s[:,1],1] - c4n[n4s[:,0],1])/2,r1D)
		c4s = np.vstack((c4sx.flatten(), c4sy.flatten())).transpose()
		c4nNew = np.vstack((c4n, c4s))

	if degree > 2:
		i_ind = np.setdiff1d(np.arange(0,nrLocal),BDindex_F.flatten())
		c4ix = np.outer(c4n[n4e[:,0],0],r[i_ind]+1)/2 + np.outer(c4n[n4e[:,1],0],s[i_ind]+1)/2 \
		   - np.outer(c4n[n4e[:,2],0], r[i_ind]+s[i_ind])/2
		c4iy = np.outer(c4n[n4e[:,0],1],r[i_ind]+1)/2 + np.outer(c4n[n4e[:,1],1],s[i_ind]+1)/2 \
		   - np.outer(c4n[n4e[:,2],1], r[i_ind]+s[i_ind])/2
		c4i = np.vstack((c4ix.flatten(), c4iy.flatten())).transpose()
		c4nNew = np.vstack((c4nNew, c4i))

	return (c4nNew, ind4e, ind4Db, ind4Nb)

def compute_n4s(n4e):
	"""
	Get a matrix whose each row contains end points of the corresponding side (or edge)

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``n4s`` (``int32 array``) : nodes for sides

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> n4s = compute_n4s(n4e)
		>>> n4s
		array([[1, 3],
		   [3, 0],
		   [1, 2],
		   [0, 1],
		   [2, 3]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind = np.unique(x + y*1.0j, return_index=True)
	n4sInd = np.sort(ind)
	n4s = allSides[n4sInd,:]
	return n4s

def compute_s4e(n4e):
	"""
	Get a matrix whose each row contains three side numbers of the corresponding element

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``s4e`` (``int32 array``) : sides for elements

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> s4e = compute_s4e(n4e)
		>>> s4e
		array([[0, 1, 3],
		   [0, 2, 4]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(3,-1).transpose().astype('int')
	return s4e

def compute_e4s(n4e):
	"""
	Get a matrix whose each row contains two elements sharing the corresponding side
	If second column is -1, the corresponding side is on the boundary

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``e4s`` (``int32 array``) : elements for sides

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> e4s = compute_e4s(n4e)
		>>> e4s
		array([[ 0,  1],
		   [ 0, -1],
		   [ 1, -1],
		   [ 0, -1],
		   [ 1, -1]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	n4sInd = np.sort(ind)

	nrElems = n4e.shape[0]
	elemNumbers = np.hstack((np.hstack((np.arange(0,nrElems),np.arange(0,nrElems))),np.arange(0,nrElems)))

	e4s=np.zeros((ind.size,2),int)
	e4s[:,0]=elemNumbers[n4sInd] + 1

	allElems4s=np.zeros(allSides.shape[0],int)
	tmp2 = np.bincount((back + 1),weights = (elemNumbers + 1))
	allElems4s[ind]=tmp2[1::]
	e4s[:,1] = allElems4s[n4sInd] - e4s[:,0]
	e4s=e4s-1
	return e4s

def refineUniformRed(c4n, n4e, n4Db, n4Nb):
	"""
	Refine a given mesh uniformly using the red refinement

	Paramters
		- ``c4n`` (``float64 array``) : coordinates for elements
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4Db`` (``int32 array``) : nodes for Difichlet boundary
		- ``n4Nb`` (``int32 array``) : nodes for Neumann boundary

	Returns
		- ``c4nNew`` (``float64 array``) : coordinates for element obtained from red refinement
		- ``n4eNew`` (``int32 array``) : nodes for element obtained from red refinement
		- ``n4DbNew`` (``int32 array``) : nodes for Dirichlet boundary obtained from red refinement
		- ``n4NbNew`` (``int32 array``) : nodes for Neumann boundary obtained from red refinement

	Example
		>>> c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> n4Db = np.array([[0, 1], [1, 2]])
		>>> n4Nb = np.array([[2, 3],[3, 0]])
		>>> c4nNew, n4eNew, n4DbNew, n4NbNew = refineUniformRed(c4n, n4e, n4Db, n4Nb)
		>>> c4nNew
		array([[ 0. ,  0. ],
		   [ 1. ,  0. ],
		   [ 1. ,  1. ],
		   [ 0. ,  1. ],
		   [ 0.5,  0.5],
		   [ 0. ,  0.5],
		   [ 1. ,  0.5],
		   [ 0.5,  0. ],
		   [ 0.5,  1. ]])
		>>> n4eNew
		array([[1, 4, 7],
		   [4, 3, 5],
		   [5, 7, 4],
		   [7, 5, 0],
		   [3, 4, 8],
		   [4, 1, 6],
		   [6, 8, 4],
		   [8, 6, 2]])
		>>> n4DbNew
		array([[0, 7],
		   [7, 1],
		   [1, 6],
		   [6, 2]])
		>>>n4NbNew
		array([[2, 8],
		   [8, 3],
		   [3, 5],
		   [5, 0]])
	"""
	nrNodes = c4n.shape[0]
	nrElems = n4e.shape[0]
	n4s = compute_n4s(n4e)
	nrSides = n4s.shape[0]
	from scipy.sparse import coo_matrix
	newNodes4s = coo_matrix((np.arange(0,nrSides)+nrNodes, (n4s[:,0], n4s[:,1])), shape=(nrNodes, nrNodes))
	newNodes4s = newNodes4s.tocsr()
	newNodes4s = newNodes4s + newNodes4s.transpose()

	mid4s = (c4n[n4s[:,0],:] + c4n[n4s[:,1],:]) * 0.5
	c4nNew = np.vstack((c4n, mid4s))

	n4eNew = np.zeros((4 * nrElems, 3), dtype=int)
	for elem in range(0,nrElems):
		nodes = n4e[elem,:]
		newNodes = np.array([newNodes4s[nodes[0],nodes[1]], newNodes4s[nodes[1],nodes[2]], newNodes4s[nodes[2],nodes[0]]])
		n4eNew[4*elem + np.arange(0,4),:] = np.array([[nodes[0], newNodes[0], newNodes[2]], 
			[newNodes[0], nodes[1], newNodes[1]], [newNodes[1], newNodes[2], newNodes[0]],
			[newNodes[2], newNodes[1], nodes[2]]])

	n4DbNew = np.zeros((2 * n4Db.shape[0], 2), dtype = int)
	for side in range(0, n4Db.shape[0]):
		nodes = n4Db[side,:]
		newNodes = newNodes4s[nodes[0], nodes[1]]
		n4DbNew[2*side + np.arange(0,2),:] = np.array([[nodes[0], newNodes], [newNodes, nodes[1]]])

	n4NbNew = np.zeros((2 * n4Nb.shape[0], 2), dtype = int)
	for side in range(0, n4Nb.shape[0]):
		nodes = n4Nb[side,:]
		newNodes = newNodes4s[nodes[0], nodes[1]]
		n4NbNew[2*side + np.arange(0,2),:] = np.array([[nodes[0], newNodes], [newNodes, nodes[1]]])

	return (c4nNew, n4eNew, n4DbNew, n4NbNew)

	


def sample():
	from os import listdir
	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve

	folder = path.join(mz.__path__[0], 'samples', 'benchmark01')
	c4n_path = [file for file in listdir(folder) if 'c4n' in file][0]
	n4e_path = [file for file in listdir(folder) if 'n4e' in file][0]
	ind4e_path = [file for file in listdir(folder) if 'idx4e' in file][0]
	n4db_path = [file for file in listdir(folder) if 'n4sDb' in file][0]

	print(c4n_path)
	print(n4e_path)
	print(ind4e_path)
	print(n4db_path)

	c4n = np.fromfile(path.join(folder, c4n_path), dtype=np.float64)
	n4e = np.fromfile(path.join(folder, n4e_path), dtype=np.int32)
	ind4e = np.fromfile(path.join(folder, ind4e_path), dtype=np.int32)
	n4db = np.fromfile(path.join(folder, n4db_path), dtype=np.int32)

	print (c4n)
	print (n4e)
	print (ind4e)
	print (n4db)

	M_R = np.array([[2, 1, 1], [1, 2, 1],  [1, 1, 2]], dtype=np.float64) / 6.
	Srr_R = np.array([[1, -1, 0], [-1, 1, 0],  [0, 0, 0]], dtype=np.float64) / 2.
	Srs_R = np.array([[1, 0, -1], [-1, 0, 1],  [0, 0, 0]], dtype=np.float64) / 2.
	Ssr_R = np.array([[1, -1, 0], [0, 0, 0],  [-1, 1 ,0]], dtype=np.float64) / 2.
	Sss_R = np.array([[1, 0, -1], [0, 0, 0],  [-1, 0, 1]], dtype=np.float64) / 2.
	Dr_R = np.array([[-1, 1, 0], [-1, 1, 0],  [-1, 1, 0]], dtype=np.float64) / 2.
	Ds_R = np.array([[-1, 0, 1], [-1, 0 ,1],  [-1, 0 ,1]], dtype=np.float64) / 2.

	dim = 2
	nrNodes = int(len(c4n) / dim)
	nrElems = int(len(n4e) / 3)
	nrLocal = int(Srr_R.shape[0])

	f = np.ones((nrLocal * nrElems), dtype=np.float64) # RHS

	print((nrNodes, nrElems, dim, nrLocal))

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)
	b = np.zeros(nrNodes)

	Poison_2D = lib['Poisson_2D_Triangle'] # need the extern!!
	Poison_2D.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
						c_void_p,
						c_void_p, c_void_p, c_void_p, c_void_p, c_int,
						c_void_p,
						c_void_p, c_void_p, c_void_p, c_void_p,)
	Poison_2D.restype = None
	Poison_2D(c_void_p(n4e.ctypes.data), c_void_p(ind4e.ctypes.data),
		c_void_p(c4n.ctypes.data), c_int(nrElems),
		c_void_p(M_R.ctypes.data),
		c_void_p(Srr_R.ctypes.data),
		c_void_p(Srs_R.ctypes.data),
		c_void_p(Ssr_R.ctypes.data),
		c_void_p(Sss_R.ctypes.data),
		c_int(nrLocal),
		c_void_p(f.ctypes.data),
		c_void_p(I.ctypes.data),
		c_void_p(J.ctypes.data),
		c_void_p(Alocal.ctypes.data),
		c_void_p(b.ctypes.data))


	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4db)

	# print STIMA_CSR

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	print(x)

	# header_str = """
	# TITLE = "Example 2D Finite Element Triangulation Plot"
	# VARIABLES = "X", "Y", "U"
	# ZONE T="P_1", DATAPACKING=POINT, NODES={0}, ELEMENTS={1}, ZONETYPE=FETRIANGLE
	# """.format(nrNodes, nrElems)
	# print(header_str)

	# data_str = ""
	# for k in range(0, nrNodes):
	# 	data_str += "{0} {1} {2}\n".format(coord_x[k], coord_y[k], u[k])

	# np.savetxt(os.join(os.getcwd(), 'sample.dat'), (n4e+1).reshape((nrElems, 3)),
	# 	fmt='%d',
	# 	header=header_str + data_str, comments="")