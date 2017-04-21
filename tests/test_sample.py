import unittest
import numpy as np
from numpy import linalg as LA
class TestBasic(unittest.TestCase):
	def test_test(self):
		self.assertTrue(True)

	def test_import(self):
		import mozart as mz
		self.assertTrue(True)

	def test_authors(self):
		import mozart as mz
		authors = ('Yoon-gu Hwang <yz0624@gmail.com>', 'Dong-Wook Shin <dwshin.yonsei@gmail.com>', 'Ji-Yeon Suh <suh91919@gmail.com>')
		self.assertEqual(mz.__author__, authors)

class TestFemCommon(unittest.TestCase):
	def test_nJacobiP(self):
		from mozart.poisson.fem.common import nJacobiP
		x = np.linspace(-1,1,5)
		P = nJacobiP(x,0,0,0)
		diff_P = P - 0.707106781186548 * np.ones(5,float)
		P2 = nJacobiP(x,0,0,1)
		diff_P2 = P2 - 1.224744871391589*x
		P3 = nJacobiP(x,0,0,2)
		diff_P3 = P3 - np.array([1.581138830084190, -0.197642353760524, -0.790569415042095, -0.197642353760524, 1.581138830084190])
		self.assertTrue(LA.norm(diff_P) < 1E-8)
		self.assertTrue(LA.norm(diff_P2) < 1E-8)
		self.assertTrue(LA.norm(diff_P3) < 1E-8)

	def test_nJacobiGQ(self):
		from mozart.poisson.fem.common import nJacobiGQ
		x0, w0 = nJacobiGQ(0,0,0)
		diff_x0 = x0 - 0.0
		diff_w0 = w0 - 2.0
		x1, w1 = nJacobiGQ(0,0,1)
		diff_x1 = x1 - np.array([-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)])
		diff_w1 = w1 - np.array([1.0, 1.0])
		x2, w2 = nJacobiGQ(0,0,2)
		diff_x2 = x2 - np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
		diff_w2 = w2 - np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
		x3, w3 = nJacobiGQ(1,1,1)
		diff_x3 = x3 - np.array([-0.447213595499958,   0.447213595499958])
		diff_w3 = w3 - np.array([0.666666666666667,   0.666666666666667])
		self.assertTrue(LA.norm(diff_x0) < 1E-8)
		self.assertTrue(LA.norm(diff_w0) < 1E-8)
		self.assertTrue(LA.norm(diff_x1) < 1E-8)
		self.assertTrue(LA.norm(diff_w1) < 1E-8)
		self.assertTrue(LA.norm(diff_x2) < 1E-8)
		self.assertTrue(LA.norm(diff_w2) < 1E-8)
		self.assertTrue(LA.norm(diff_x3) < 1E-8)
		self.assertTrue(LA.norm(diff_w3) < 1E-8)

	def test_nJacobiGL(self):
		from mozart.poisson.fem.common import nJacobiGL
		x0 = nJacobiGL(0,0,0)
		diff_x0 = x0 - 0.0;
		x1 = nJacobiGL(0,0,1)
		diff_x1 = x1 - np.array([-1.0, 1.0])
		x2 = nJacobiGL(0,0,2)
		diff_x2 = x2 - np.array([-1.0, 0.0, 1.0])
		self.assertTrue(LA.norm(diff_x0) < 1E-8)
		self.assertTrue(LA.norm(diff_x1) < 1E-8)
		self.assertTrue(LA.norm(diff_x2) < 1E-8)

	def test_DnJacobiP(self):
		from mozart.poisson.fem.common import DnJacobiP
		x = np.linspace(-1,1,5)
		dP = DnJacobiP(x,0,0,0)
		diff_dP = dP - np.zeros(5,float)
		dP2 = DnJacobiP(x,0,0,1)
		diff_dP2 = dP2 - 1.224744871391589*np.ones(5,float)
		self.assertTrue(LA.norm(diff_dP) < 1E-8)
		self.assertTrue(LA.norm(diff_dP2) < 1E-8)

	def test_VandermondeM1D(self):
		from mozart.poisson.fem.common import VandermondeM1D
		r = np.array([-1.0, 1.0])
		V1D = VandermondeM1D(1,r)
		diff_V1D = V1D - np.array([[0.707106781186548, -1.224744871391589], [0.707106781186548, 1.224744871391589]])
		self.assertTrue(LA.norm(diff_V1D) < 1E-8)

	def test_DVandermondeM1D(self):
		from mozart.poisson.fem.common import DVandermondeM1D
		r = np.array([-1.0, 1.0])
		DVr = DVandermondeM1D(1,r)
		diff_DVr = DVr - np.array([[0.0, 1.224744871391589], [0.0, 1.224744871391589]])
		self.assertTrue(LA.norm(diff_DVr) < 1E-8)

	def test_Dmatrix1D(self):
		from mozart.poisson.fem.common import Dmatrix1D, VandermondeM1D
		r = np.array([-1.0, 1.0])
		V = VandermondeM1D(1,r)
		Dr = Dmatrix1D(1,r,V)
		diff_Dr = Dr - np.array([[-0.5, 0.5], [-0.5, 0.5]])
		self.assertTrue(LA.norm(diff_Dr) < 1E-8)

	def test_RefNodes_Tri(self):
		from mozart.poisson.fem.common import RefNodes_Tri
		N = 0
		r0, s0 = RefNodes_Tri(N)
		diff_r0 = r0 - np.array([-1.0/3])
		diff_s0 = s0 - np.array([-1.0/3])
		N = 3
		r, s = RefNodes_Tri(N)
		diff_r = r - np.array([-1, -1.0/3, 1.0/3, 1, -1, -1.0/3, 1.0/3, -1, -1.0/3, -1])
		diff_s = s - np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
		self.assertTrue(LA.norm(diff_r) < 1E-8)
		self.assertTrue(LA.norm(diff_s) < 1E-8)

	def test_rs2ab(self):
		from mozart.poisson.fem.common import RefNodes_Tri, rs2ab
		N = 3
		r, s = RefNodes_Tri(N)
		a, b = rs2ab(r,s)
		diff_a = a - np.array([-1, -1.0/3, 1.0/3, 1, -1, 0, 1, -1, 1, -1])
		diff_b = b - np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
		self.assertTrue(LA.norm(diff_a) < 1E-8)
		self.assertTrue(LA.norm(diff_b) < 1E-8)

	def test_Simplex2DP(self):
		from mozart.poisson.fem.common import Simplex2DP
		a = np.array([0,1])
		b = np.array([2,3])
		p = Simplex2DP(a,b,0,0)
		diff_p = p - np.array([ 0.70710678, 0.70710678])
		self.assertTrue(LA.norm(diff_p) < 1E-8)

	def test_Vandermonde2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D
		N = 2
		r, s = RefNodes_Tri(N)
		V2D = Vandermonde2D(N,r,s)
		diff_V2D = V2D - np.array([[ 0.707106781186548,  -1.              ,   1.224744871391590,  -1.732050807568878,   2.121320343559643,   2.738612787525831],
			[ 0.707106781186548,  -1.              ,   1.224744871391590,                   0,                   0,  -1.369306393762915],
			[ 0.707106781186548,  -1.              ,   1.224744871391590,   1.732050807568878,  -2.121320343559643,   2.738612787525831],
			[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,  -0.866025403784439,  -1.590990257669732,   0.684653196881458],
			[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,   0.866025403784439,   1.590990257669732,   0.684653196881458],
			[ 0.707106781186548,   2.000000000000001,   3.674234614174769,                   0,                   0,                   0]])
		self.assertTrue(LA.norm(diff_V2D) < 1E-8)

	def test_GradSimplex2DP(self):
		from mozart.poisson.fem.common import RefNodes_Tri, rs2ab, GradSimplex2DP
		N = 2
		r, s = RefNodes_Tri(N)
		a, b = rs2ab(r,s)
		dmodedr, dmodeds = GradSimplex2DP(a,b,1,1)
		diff_dmodedr = dmodedr - np.array([-2.121320343559642, -2.121320343559642, -2.121320343559642,
			 3.181980515339464,  3.181980515339464,  8.485281374238570])
		diff_dmodeds = dmodeds - np.array([-6.363961030678929, -1.060660171779821,  4.242640687119286,
			-1.060660171779822,  4.242640687119286,  4.242640687119285])
		self.assertTrue(LA.norm(diff_dmodedr) < 1E-8)
		self.assertTrue(LA.norm(diff_dmodeds) < 1E-8)

	def test_GradVandermonde2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, GradVandermonde2D
		N = 2
		r, s = RefNodes_Tri(N)
		V2Dr, V2Ds = GradVandermonde2D(N,r,s)
		diff_V2Dr = V2Dr - np.array([[ 0.                 ,  0.                 ,  0.                 ,
		      1.732050807568877, -2.121320343559642, -8.215838362577491],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877, -2.121320343559642,  0.               ],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877, -2.121320343559642,  8.215838362577491],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  3.181980515339464, -4.107919181288746],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  3.181980515339464,  4.107919181288746],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  8.485281374238570,  0.               ]])
		diff_V2Ds = V2Ds - np.array([[0.               ,  1.5              , -4.898979485566358,
			  0.866025403784439, -6.363961030678929, -2.738612787525831],
			[ 0.               ,  1.5              , -4.898979485566358,
			  0.866025403784439, -1.060660171779821,  1.369306393762915],
			[ 0.               ,  1.500000000000000, -4.898979485566358,
			  0.866025403784439,  4.242640687119286,  5.477225575051659],
			[ 0.               ,  1.500000000000000,  1.224744871391589,
			  0.866025403784439, -1.060660171779822, -1.369306393762915],
			[ 0.               ,  1.500000000000000,  1.224744871391589,
			  0.866025403784439,  4.242640687119286,  2.738612787525830],
			[ 0.               ,  1.500000000000000,  7.348469228349536,
			  0.866025403784439,  4.242640687119285,  0.               ]])
		self.assertTrue(LA.norm(diff_V2Dr) < 1E-8)
		self.assertTrue(LA.norm(diff_V2Ds) < 1E-8)

	def test_Dmatrices2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D, Dmatrices2D
		N = 2
		r, s = RefNodes_Tri(N)
		V = Vandermonde2D(N,r,s)
		Dr, Ds = Dmatrices2D(N,r,s,V)
		diff_Dr = Dr - np.array([[-1.5,  2., -0.5,  0.,  0.,  0.],
		    [-0.5,  0.,  0.5,  0.,  0.,  0.],
		    [ 0.5, -2.,  1.5,  0.,  0.,  0.],
		    [-0.5,  1., -0.5, -1.,  1.,  0.],
		    [ 0.5, -1.,  0.5, -1.,  1.,  0.],
		    [ 0.5,  0., -0.5, -2.,  2.,  0.]])            
		diff_Ds = Ds - np.array([[-1.5,  0.,  0.,  2.,  0.,  -0.5],
			[-0.5, -1.,  0.,  1.,  1., -0.5],
			[ 0.5, -2.,  0.,  0.,  2., -0.5],
			[-0.5,  0.,  0.,  0.,  0.,  0.5],
			[ 0.5, -1.,  0., -1.,  1.,  0.5],
			[ 0.5,  0.,  0., -2.,  0.,  1.5]])
		self.assertTrue(LA.norm(diff_Dr) < 1E-8)
		self.assertTrue(LA.norm(diff_Ds) < 1E-8)

class TestFemInterval(unittest.TestCase):
	def test_1d_uniform_interval(self):
		from mozart.mesh.rectangle import interval
		a, b, M, N = (0, 1, 4, 2)
		c4n, n4e, n4db, ind4e = interval(a,b,M,N)
		diff_c4n = c4n - np.linspace(a,b,M*N+1)
		diff_n4e = n4e - np.array([[0,2], [2,4], [4,6], [6,8]])
		diff_n4db = n4db - np.array([0, 8])
		diff_ind4e = ind4e - np.array([[0,1,2], [2,3,4], [4,5,6], [6,7,8]])
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)
		self.assertTrue(LA.norm(diff_n4db) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4e) < 1E-8)

	def test_getPoissonMatrix1D(self):
		from mozart.poisson.fem.interval import getMatrix
		M1, S1, D1 = getMatrix(1)
		diff_M1 = M1 - np.array([[ 2,  1], [ 1, 2]]) / 3.
		diff_S1 = S1 - np.array([[ 1, -1], [-1, 1]]) / 2.
		diff_D1 = D1 - np.array([[-1,  1], [-1, 1]]) / 2.
		M2, S2, D2 = getMatrix(2)
		diff_M2 = M2 - np.array([[ 4,  2, -1], [ 2, 16,  2], [-1,  2, 4]]) / 15.
		diff_S2 = S2 - np.array([[ 7, -8,  1], [-8, 16, -8], [ 1, -8, 7]]) / 6.
		diff_D2 = D2 - np.array([[-3,  4, -1], [-1,  0,  1], [ 1, -4, 3]]) / 2.
		self.assertTrue(LA.norm(diff_M1) < 1E-8)
		self.assertTrue(LA.norm(diff_S1) < 1E-8)
		self.assertTrue(LA.norm(diff_D1) < 1E-8)
		self.assertTrue(LA.norm(diff_M2) < 1E-8)
		self.assertTrue(LA.norm(diff_S2) < 1E-8)
		self.assertTrue(LA.norm(diff_D2) < 1E-8)

	def test_solve(self):
		from mozart.mesh.rectangle import interval
		N = 3
		c4n, n4e, n4db, ind4e = interval(0, 1, 4, N)
		f = lambda x: np.ones_like(x)
		u_D = lambda x: np.zeros_like(x)
		from mozart.poisson.fem.interval import solve
		x = solve(c4n, n4e, n4db, ind4e, f, u_D, N)
		diff_x = x - np.array([                 0,   0.038194444444444,   0.069444444444444,   0.093749999999999,   0.111111111111110,
		   0.121527777777777,   0.124999999999999,   0.121527777777777,   0.111111111111110,   0.093749999999999,   0.069444444444444,
		   0.038194444444444,                   0])
		self.assertTrue(LA.norm(diff_x) < 1E-8)

	def test_computeError(self):
		from mozart.mesh.rectangle import interval
		from mozart.poisson.fem.interval import solve
		from mozart.poisson.fem.interval import computeError
		N = 2
		iter = 4
		f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
		u_D = lambda x: np.zeros_like(x)
		exact_u = lambda x: np.sin(np.pi * x)
		exact_ux = lambda x: np.pi * np.cos(np.pi * x)
		L2error = np.zeros(iter, dtype = np.float64)
		sH1error = np.zeros(iter, dtype = np.float64)
		h = np.zeros(iter, dtype = np.float64)
		for j in range(0,iter):
			c4n, n4e, n4db, ind4e = interval(0,1,2**(j+1),N)
			x = solve(c4n, n4e, n4db, ind4e, f, u_D, N)
			L2error[j], sH1error[j] = computeError(c4n, n4e, ind4e, exact_u, exact_ux, x, N, N+3)
			h[j] = 1 / 2.0**(j+1)
		rateL2=(np.log(L2error[1:])-np.log(L2error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
		rateH1=(np.log(sH1error[1:])-np.log(sH1error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
		self.assertTrue(np.abs(rateL2[-1]) > N+0.9)
		self.assertTrue(np.abs(rateH1[-1]) > N-0.1)

class TestFemTriangle(unittest.TestCase):
	def test_getMatrix2D(self):
		from mozart.poisson.fem.triangle import getMatrix
		N = 1
		M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R, M1D_R = getMatrix(N)
		diff_M = M_R - np.array([[1.0/3, 1.0/6, 1.0/6,], [1.0/6, 1.0/3, 1.0/6], [1.0/6, 1.0/6, 1.0/3]], dtype = np.float64)
		diff_Srr = Srr_R - np.array([[1.0/2, -1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [0.0, 0.0, 0.0]], dtype = np.float64)
		diff_Srs = Srs_R - np.array([[1.0/2, 0.0, -1.0/2], [-1.0/2, 0.0, 1.0/2], [0.0, 0.0, 0.0]], dtype = np.float64)
		diff_Ssr = Ssr_R - np.array([[1.0/2, -1.0/2, 0.0], [0.0, 0.0, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
		diff_Sss = Sss_R - np.array([[1.0/2, 0.0, -1.0/2], [0.0, 0.0, 0.0], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
		diff_Dr = Dr_R - np.array([[-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
		diff_Ds = Ds_R - np.array([[-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
		diff_M1D = M1D_R - np.array([[2.0/3, 1.0/3], [1.0/3, 2.0/3]])
		self.assertTrue(LA.norm(diff_M) < 1E-8)
		self.assertTrue(LA.norm(diff_Srr) < 1E-8)
		self.assertTrue(LA.norm(diff_Srs) < 1E-8)
		self.assertTrue(LA.norm(diff_Ssr) < 1E-8)
		self.assertTrue(LA.norm(diff_Sss) < 1E-8)
		self.assertTrue(LA.norm(diff_Dr) < 1E-8)
		self.assertTrue(LA.norm(diff_Ds) < 1E-8)
		self.assertTrue(LA.norm(diff_M1D) < 1E-8)

	def test_compute_n4s(self):
		from mozart.poisson.fem.triangle import compute_n4s
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		n4s = compute_n4s(n4e)
		diff_n4s = n4s - np.array([[1, 3], [2, 4], [4, 6], [5, 7], [3, 0], [1, 4], [2, 5],
			[6, 3], [4, 7], [5, 8], [0, 1], [4, 3], [1, 2], [5, 4], [7, 6], [8, 7]])
		self.assertTrue(LA.norm(diff_n4s) < 1E-8)

	def test_compute_s4e(self):
		from mozart.poisson.fem.triangle import compute_s4e
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		s4e = compute_s4e(n4e)
		diff_s4e = s4e - np.array([[0, 4, 10], [0, 5, 11], [1, 5, 12], [1, 6, 13],
			[2, 7, 11], [2, 8, 14], [3, 8, 13], [3, 9, 15]])
		self.assertTrue(LA.norm(diff_s4e) < 1E-8)

	def test_compute_e4s(self):
		from mozart.poisson.fem.triangle import compute_e4s
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		e4s = compute_e4s(n4e)
		diff_e4s = e4s - np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, -1], [1, 2],
			[3, -1], [4, -1], [5, 6], [7, -1], [0, -1], [1, 4], [2, -1], [3, 6], [5, -1], [7, -1]])
		self.assertTrue(LA.norm(diff_e4s) < 1E-8)

	def test_refineUniformRed(self):
		from mozart.poisson.fem.triangle import refineUniformRed
		c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]])
		n4e = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
		n4Db = np.array([[0, 1], [1, 2]])
		n4Nb = np.array([[2, 3], [3, 0]])
		c4n, n4e, n4Db, n4Nb = refineUniformRed(c4n, n4e, n4Db, n4Nb)
		diff_c4n = c4n - np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5], [0.5, 0.],
			[1., 0.5], [0.5, 1.], [0., 0.5], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]])
		diff_n4e = n4e - np.array([[0, 5, 12], [5, 1, 9], [9, 12, 5], [12, 9, 4], [1, 6, 9], [6, 2, 10], [10, 9, 6],
			[9, 10, 4], [2, 7, 10], [7, 3, 11], [11, 10, 7], [10, 11, 4], [3, 8, 11], [8, 0, 12], [12, 11, 8], [11, 12, 4]])
		diff_n4Db = n4Db - np.array([[0, 5], [5, 1], [1, 6], [6, 2]])
		diff_n4Nb = n4Nb - np.array([[2, 7], [7, 3], [3, 8], [8, 0]])
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)
		self.assertTrue(LA.norm(diff_n4Db) < 1E-8)
		self.assertTrue(LA.norm(diff_n4Nb) < 1E-8)

	def test_getIndex(self):
		from mozart.poisson.fem.triangle import getIndex
		N = 3
		c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
		n4e = np.array([[1, 3, 0], [3, 1, 2]])
		n4sDb = np.array([[0, 1], [2, 3], [3, 4]])
		n4sNb = np.array([[1, 2]])
		c4nNew, ind4e, ind4Db, ind4Nb = getIndex(N, c4n, n4e, n4sDb, n4sNb)
		diff_c4nNew = c4nNew - np.array([[0., 0.], [1.,  0.], [1., 1.], [ 0., 1.], [2.0/3, 1.0/3],
			[1.0/3, 2.0/3], [0., 2.0/3], [0., 1.0/3], [1., 1.0/3], [1., 2.0/3], [1.0/3, 0.],
			[2.0/3, 0.], [2.0/3, 1.], [1.0/3, 1.], [1.0/3, 1.0/3], [2.0/3, 2.0/3]])
		diff_ind4e = ind4e - np.array([[ 0, 10, 11,  1,  7, 14,  4,  6,  5,  3],
			[ 2, 12, 13,  3,  9, 15,  5,  8,  4,  1]])
		diff_ind4Db = ind4Db - np.array([ 0,  1,  2,  3,  4,  4,  5, 10, 11, 12, 13])
		diff_ind4Nb = ind4Nb - np.array([[1, 8, 9, 2]])
		self.assertTrue(LA.norm(diff_c4nNew) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4e) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4Db) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4Nb) < 1E-8)


class TestTecplot(unittest.TestCase):
	def test_tecplot_triangle(self):
		from mozart.common.etc import tecplot_triangle
		c4n = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
		n4e = np.array([[0, 1, 2]])
		u = np.array([10, 20, 30])
		tecplot_triangle('sample.dat', c4n, n4e, u)
		self.assertTrue(True)

class TestCommonModuleMethods(unittest.TestCase):
	def test_prefix_by_os(self):
		answer_sheet ={"linux" : "linux", "linux32" : "linux", 
			"darwin" : "osx", "win32" : "win64"}

		from mozart.common.etc import prefix_by_os
		for case, answer in answer_sheet.items():
			res = prefix_by_os(case)
			print(case, answer, res)
			self.assertEqual(res, answer)

	def test_benchmark01_sample(self):
		from mozart.poisson.fem.triangle import sample
		sample()
		self.assertTrue(True)