from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np

# OS Detection Code
prefix = "linux"
if platform == "linux" or platform == "linux32":
	prefix = "linux"
elif platform == "darwin":
	prefix = "osx"
elif platform == "win32":
	prefix = "win64"

dllpath = path.join(mz.__path__[0], prefix + '_' + 'libmozart.so')
lib = CDLL(dllpath)

def getMatrix1D(degree):
	"""
	Get FEM matrices on the reference domain I = [-1, 1]

	Paramters
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference domain
		- ``S_R`` (``float64 array``) : Stiffness matrix on the reference domain
		- ``D_R`` (``float64 array``) : Differentiation matrix on the reference domain

	"""

	r = np.linspace(-1, 1, degree+1)
	V = VandermondeM1D(degree, r)
	invV = np.linalg.inv(V)
	M_R = np.dot(np.transpose(invV),invV)
	D_R = Dmatrix1D(degree, r, V)
	S_R = np.dot(np.dot(np.transpose(D_R),M_R),D_R)
	return (M_R, S_R, D_R)

def nJacobiP(x, alpha=0, beta=0, degree=0):
	"""
	Evaluate normalized Jacobi polynomial of type alpha, beta > -1 at point x for order n
	(the special case of alpha = beta = 0, knows as the normalized Legendre polynomial)

	Paramters
		- ``x`` (``float64 array``) : variable x
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``P`` (``float64 array``) : the value of degree-th order normalized Jacobi polynomial at x
	
	Example
		>>> N = 2
		>>> x = np.array([-1, 0, 1])
		>>> from mozart.poisson.solve import nJacobiP
		>>> P = nJacobiP(x,0,0,N)
		>>> print(P)
		array([ 1.58113883, -0.79056942,  1.58113883])	
	"""
	Pn = np.zeros((degree+1,x.size),float)
	Pn[0,:] = np.sqrt(2.0**(-alpha-beta-1) * np.math.gamma(alpha+beta+2) / ((np.math.gamma(alpha + 1) * np.math.gamma(beta + 1))))

	if degree == 0:
		P = Pn
	else:
		Pn[1,:] = np.multiply(Pn[0,:]*np.sqrt((alpha+beta+3.0)/((alpha+1)*(beta+1))),((alpha+beta+2)*x+(alpha-beta)))/2
		a_n = 2.0/(2+alpha+beta)*np.sqrt((alpha+1.0)*(beta+1.0)/(alpha+beta+3.0))
		for n in range(2,degree+1):
			anew=2.0/(2*n+alpha+beta)*np.sqrt(n*(n+alpha+beta)*(n+alpha)*(n+beta)/((2*n+alpha+beta-1.0)*(2*n+alpha+beta+1.0)))
			b_n=-(alpha**2-beta**2)/((2*(n-1)+alpha+beta)*(2*(n-1)+alpha+beta+2.0))
			Pn[n,:]=(np.multiply((x-b_n),Pn[n-1,:])-a_n*Pn[n-2,:])/anew
			a_n=anew

	P = Pn[degree,:]
	return P

def DnJacobiP(x, alpha=0, beta=0, degree=0):
	"""
	Evaluate the derivative of the normalized Jacobi polynomial of type alpha, beta > -1 at point x for order n

	Paramters
		- ``x`` (``float64 array``) : variable x
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``dP`` (``float64 array``) : the value of the derivative of the normalized Jacobi polynomial at x according to alpha, beta, degree
	
	Example
		>>> N = 2
		>>> x = np.array([-1, 0, 1])
		>>> from mozart.poisson.solve import DnJacobiP
		>>> dP = DnJacobiP(x,0,0,N)
		>>> print(dP)
		array([-4.74341649,  0.        ,  4.74341649])	
	"""
	dP = np.zeros(x.size,float)
	if degree == 0:
		dP[:] = 0
	else:
		dP[:] = np.sqrt(degree*(degree+alpha+beta+1.0))*nJacobiP(x,alpha+1,beta+1,degree-1)
	return dP

def nJacobiGQ(alpha=0, beta=0, degree=0):
	"""
	Compute the degree-th order Gauss quadrature points x and weights w
	associated with the nomalized Jacobi polynomial of type alpha, beta > -1

	Paramters
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : Gauss quadrature points
		- ``w`` (``float64 array``) : Gauss quadrature weights
	
	Example
		>>> N = 2
		>>> from mozart.poisson.solve import nJacobiGQ
		>>> x, w = nJacobiGQ(0,0,N)
		>>> print(x)
		array([ -7.74596669e-01,  -4.78946310e-17,   7.74596669e-01])
		>>> print(w)
		array([ 0.55555556,  0.88888889,  0.55555556])
	"""
	if degree == 0:
		x = -(alpha - beta)/(alpha + beta + 2.0)
		w = 2
	else:
		if alpha + beta < 10*np.finfo(float).eps:
			tmp = np.zeros(degree+1)
			tmp[1:] = -(alpha**2-beta**2)/((2.0*np.arange(1,degree+1)+alpha+beta+2.0)*(2.0*np.arange(1,degree+1)+alpha+beta))/2.0
			J = np.diag(tmp) + np.diag(2.0/(2.0*np.arange(1,degree+1)+alpha+beta)*np.sqrt(np.arange(1,degree+1)*(np.arange(1,degree+1)+alpha+beta)* \
					(np.arange(1,degree+1)+alpha)*(np.arange(1,degree+1)+beta)/(2.0*np.arange(1,degree+1)+alpha+beta-1.0)/(2.0*np.arange(1,degree+1)+alpha+beta+1.0)),1)
		else:
			J = np.diag(-(alpha**2-beta**2)/((2.0*np.arange(0,degree+1)+alpha+beta+2.0)*(2.0*np.arange(0,degree+1)+alpha+beta))/2.0)+ \
					np.diag(2.0/(2.0*np.arange(1,degree+1)+alpha+beta)*np.sqrt(np.arange(1,degree+1)*(np.arange(1,degree+1)+alpha+beta)* \
					(np.arange(1,degree+1)+alpha)*(np.arange(1,degree+1)+beta)/(2.0*np.arange(1,degree+1)+alpha+beta-1.0)/(2.0*np.arange(1,degree+1)+alpha+beta+1.0)),1)
		
		J = J + np.transpose(J)

		x, V = np.linalg.eig(J)
		w = np.transpose(V[0])**2 * 2**(alpha + beta + 1) / (alpha + beta + 1.0) * np.math.gamma(alpha + 1.0)  * \
			np.math.gamma(beta + 1.0) / np.math.gamma(alpha + beta + 1.0)
		ind = np.argsort(x)
		x = x[ind]
		w = w[ind]
	return (x, w)

def nJacobiGL(alpha=0, beta=0, degree=0):
	"""
	Compute the degree-th order Gauss Lobatto quadrature points x
	associated with the nomalized Jacobi polynomial of type :math:`\\alpha, \\beta > -1`

	Paramters
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : Gauss Lobatto quadrature points
	
	Example
		>>> N = 3
		>>> from mozart.poisson.solve import nJacobiGL
		>>> x = nJacobiGQ(0,0,N)
		>>> print(x)
		array([-1.       , -0.4472136,  0.4472136,  1.       ])
	"""
	if degree == 0:
		x = 0
	elif degree == 1:
		x = np.array([-1, 1])
	else:
		xint, w = nJacobiGQ(alpha+1,beta+1,degree-2)
		x = np.hstack((np.array([-1]),xint))
		x = np.hstack((x,np.array([1])))	
	return x

def VandermondeM1D(degree,r):
	"""
	Initialize the 1D Vandermonde matrix, :math:`V_{i,j} = \\phi_j(r_i)`

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``V1D`` (``float64 array``) : 1D Vandermonde matrix
	
	Example
		>>> N = 3
		>>> from mozart.poisson.solve import VandermondeM1D
		>>> r = np.linspace(-1,1,N+1)
		>>> V1D = VandermondeM1D(N,r)
		>>> print(V1D)
		array([[ 0.70710678, -1.22474487,  1.58113883, -1.87082869],
		[ 0.70710678, -0.40824829, -0.52704628,  0.76218947],
		[ 0.70710678,  0.40824829, -0.52704628, -0.76218947],
		[ 0.70710678,  1.22474487,  1.58113883,  1.87082869]])
	"""
	V1D = np.zeros((r.size,degree+1),float)
	for j in range(0,degree+1):
		V1D[:,j] = nJacobiP(r,0,0,j)
	return V1D

def DVandermondeM1D(degree, r):
	"""
	Initialize the derivative of modal basis (i) at (r) at order degree

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``DVr`` (``float64 array``) : Differentiate Vandermonde matrix
	
	Example
		>>> N = 3
		>>> from mozart.poisson.solve import VandermondeM1D
		>>> r = np.linspace(-1,1,N+1)
		>>> DVr = DVandermondeM1D(N,r)
		>>> print(DVr)
		array([[  0.        ,   1.22474487,  -4.74341649,  11.22497216],
		[  0.        ,   1.22474487,  -1.58113883,  -1.24721913],
		[  0.        ,   1.22474487,   1.58113883,  -1.24721913],
		[  0.        ,   1.22474487,   4.74341649,  11.22497216]])
	"""
	DVr = np.zeros((r.size,degree+1), float)
	for j in range(0,degree+1):
		DVr[:,j] = DnJacobiP(r,0,0,j)
	return DVr

def Dmatrix1D(degree, r, V):
	"""
	Initialize the derivative of modal basis (i) at (r) at order degree

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``DVr`` (``float64 array``) : Differentiate Vandermonde matrix
	
	Example
		>>> N = 3
		>>> from mozart.poisson.solve import Dmatrix1D
		>>> r = np.linspace(-1,1,N+1)
		>>> Dr = Dmatirx1D(N,r)
		>>> print(Dr)
		array([[-2.75,  4.5 , -2.25,  0.5 ],
		[-0.5 , -0.75,  1.5 , -0.25],
		[ 0.25, -1.5 ,  0.75,  0.5 ],
		[-0.5 ,  2.25, -4.5 ,  2.75]])
	"""
	Vr = DVandermondeM1D(degree, r)
	Dr = np.linalg.solve(np.transpose(V),np.transpose(Vr))
	Dr = np.transpose(Dr)
	return Dr

def one_dim_p(c4n,n4e,n4db,ind4e,f,u_D,degree):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4db`` (``int32 array``) : nodes for Dirichlet boundary
		- ``ind4e`` (``int32 array``) : indices for elements 
		- ``f`` (``lambda``) : source term 
		- ``u_D`` (``lambda``) : Dirichlet boundary condition
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : solution

	Example
		>>> N = 2
		>>> from mozart.mesh.rectangle import interval 
		>>> c4n, n4e, n4db, ind4e = interval(0, 1, 4, 2)
		>>> f = lambda x: np.ones_like(x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.solve import one_dim_p
		>>> x = one_dim_p(c4n, n4e, n4db, ind4e, f, u_D, N)
		>>> x
		array([ 0.       ,  0.0546875,  0.09375  ,  0.1171875,  0.125    ,
		   0.1171875,  0.09375  ,  0.0546875,  0.       ])
	"""
	nrLocal = degree + 1
	nrElems = n4e.shape[0]
	nrNodes = c4n.shape[0]
	Alocal = np.zeros((nrLocal * nrLocal * nrElems), dtype=np.float64)
	b = np.zeros(nrNodes, dtype=np.float64)

	from mozart.poisson.solve import getMatrix1D
	M_R, S_R, D_R = getMatrix1D(degree)
	for j in range(0,nrElems):
		Jacobi = (c4n[n4e[j,1]] - c4n[n4e[j,0]])/2.0
		Alocal[np.arange(j*(nrLocal*nrLocal),(j+1)*(nrLocal*nrLocal),1)] = S_R.flatten()/Jacobi
		b[ind4e[j]] += Jacobi * np.dot(M_R, f(c4n[ind4e[j]].flatten()))

	import numpy.matlib
	J = np.matlib.repmat(ind4e,1,nrLocal)
	J = J.flatten()
	I = ind4e.flatten()
	I = np.transpose(np.matlib.repmat(I,nrLocal,1))
	I = I.flatten()

	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve
	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4db)
	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return x

def computeError_one_dim(c4n, n4e, ind4e, exact_u, exact_ux, approx_u, degree, degree_i):
	"""
	Computes L^2-error and semi H^1-error between exact solution and approximate solution.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``ind4e`` (``int32 array``) : indices for elements
		- ``exact_u`` (``lambda``) : exact solution
		- ``exact_ux`` (``lambda``) : derivative of exact solution 
		- ``approx_u`` (``float64 array``) : approximate solution
		- ``degree`` (``int32``) : Polynomial degree
		- ``degree_i`` (``int32``) : Polynomial degree for interpolation

	Returns
		- ``L2error`` (``float64``) : L^2 error between exact solution and approximate solution.
		- ``sH1error`` (``float64``) : semi H^1 error between exact solution and approximate solution.

	Example
		>>> N = 2
		>>> from mozart.mesh.rectangle import interval 
		>>> c4n, n4e, n4db, ind4e = interval(0, 1, 4, 2)
		>>> f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.solve import one_dim_p
		>>> x = one_dim_p(c4n, n4e, n4db, ind4e, f, u_D, N)
		>>> from mozart.poisson.solution import computeError_one_dim
		>>> exact_u = lambda x: np.sin(np.pi * x)
		>>> exact_ux = lambda x: np.pi * np.cos(np.pi * x)
		>>> L2error, sH1error = computeError_one_dim(c4n, n4e, ind4e, exact_u, exact_ux, x, N, N+3)
		>>> L2error
		0.0020225729623142077
		>>> sH1error
		0.05062779815975444
	"""
	L2error = 0
	sH1error = 0

	r = np.linspace(-1, 1, degree + 1)
	V = VandermondeM1D(degree, r)
	Dr = Dmatrix1D(degree, r, V)

	r_i = np.linspace(-1, 1, degree_i + 1)
	V_i = VandermondeM1D(degree_i, r_i)
	invV_i = np.linalg.inv(V_i)
	M_R = np.dot(np.transpose(invV_i), invV_i)
	PM = VandermondeM1D(degree, r_i)
	interpM = np.transpose(np.linalg.solve(np.transpose(V), np.transpose(PM)))

	for j in range(0,n4e.shape[0]):
		Jacobi = (c4n[n4e[j,1]] - c4n[n4e[j,0]])/2.0
		approx_u_i = np.dot(interpM, approx_u[ind4e[j]])
		Dapprox_u = np.dot(Dr, approx_u[ind4e[j]]) / Jacobi
		Dapprox_u_i = np.dot(interpM, Dapprox_u)

		nodes = (1-r_i)/2*c4n[n4e[j,0]]+(1+r_i)/2*c4n[n4e[j,1]]
		diff_u = exact_u(nodes) - approx_u_i
		diff_Du = exact_ux(nodes) - Dapprox_u_i
		L2error += Jacobi*np.dot(np.dot(np.transpose(diff_u),M_R),diff_u)
		sH1error += Jacobi*np.dot(np.dot(np.transpose(diff_Du),M_R),diff_Du)

	L2error = np.sqrt(L2error)
	sH1error = np.sqrt(sH1error)
	return (L2error, sH1error)




def one_dim(c4n, n4e, n4Db, f, u_D, degree = 1):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4Db`` (``int32 array``) : Dirichlet boundary nodes
		- ``f`` (``lambda``) : source term 
		- ``u_D`` (``lambda``) : Dirichlet boundary condition
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : solution

	Example
		>>> N = 3
		>>> c4n, n4e = unit_interval(N)
		>>> n4Db = [0, N-1]
		>>> f = lambda x: np.ones_like(x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.solve import one_dim
		>>> x = one_dim(c4n, n4e, n4Db, f, u_D)
		>>> print(x)
		array([ 0.   ,  0.125,  0.   ])
	"""
	from mozart.poisson.solve import getMatrix1D
	M_R, S_R, D_R = getMatrix1D(degree)
	fval = f(c4n[n4e].flatten())
	nrNodes = int(c4n.shape[0])
	nrElems = int(n4e.shape[0])
	nrLocal = int(M_R.shape[0])

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)

	b = np.zeros(nrNodes)
	Poison_1D = lib['Poisson_1D'] # need the extern!!
	Poison_1D.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
	                    c_void_p, c_void_p, c_int,
	                    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,)
	Poison_1D.restype = None
	Poison_1D(c_void_p(n4e.ctypes.data), c_void_p(n4e.ctypes.data),
	    c_void_p(c4n.ctypes.data), c_int(nrElems),
	    c_void_p(M_R.ctypes.data),
	    c_void_p(S_R.ctypes.data),
	    c_int(nrLocal),
	    c_void_p(fval.ctypes.data),
	    c_void_p(I.ctypes.data),
	    c_void_p(J.ctypes.data),
	    c_void_p(Alocal.ctypes.data),
	    c_void_p(b.ctypes.data))
	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve
	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4Db)

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return x

def RefNodes_Tri(degree):
	"""
	Computes uniform nodes in the reference triangle for arbitrary polynomial degrees
	
	Parameters
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``r`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Example
		>>> N = 3
		>>> r, s = RefNodes_Tri(N)
		>>> r
		array([-1.        , -0.33333333,  0.33333333,  1.        , -1.        ,
		   -0.33333333,  0.33333333, -1.        , -0.33333333, -1.        ])
		>>> s
		array([-1.        , -1.        , -1.        , -1.        , -0.33333333,
		   -0.33333333, -0.33333333,  0.33333333,  0.33333333,  1.        ])
	"""
	nrLocal = int((degree + 1)*(degree + 2)/2)
	x = np.linspace(-1, 1, degree + 1)
	r = np.zeros(nrLocal, dtype = np.float64)
	s = np.zeros(nrLocal, dtype = np.float64)
	for j in range (0, degree+1):
		r[int((degree + 1)*j - j*(j-1)/2) + np.arange(0,degree+1-j,1)] = x[np.arange(0,degree+1-j,1)]
		s[int((degree + 1)*j - j*(j-1)/2) + np.arange(0,degree+1-j,1)] = x[j]
	return (r,s)

def two_dim(c4n, n4e, n4sDb, f):
	print("two_dim is called.")

def three_dim(c4n, n4e, n4sDb, f):
	print("trhee_dim is called.")

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