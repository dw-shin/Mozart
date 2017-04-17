import numpy as np
import numpy.testing as npt

def test_3d_uniform_cube_degree_one():
	from mozart.mesh.cube import cube
	x1, x2, y1, y2, z1, z2, Mx, My, Mz, N = (0, 1, 0, 1, 0, 1, 2, 2, 2, 1)
	c4n, ind4e, n4e, n4Db = cube(x1,x2,y1,y2,z1,z2,Mx,My,Mz,N)
	ref_c4n = np.matrix('[0 0 0;0.5 0 0;1 0 0;0 0.5 0;0.5 0.5 0;1 0.5 0;0 1 0;0.5 1 0;1 1 0;0 0 0.5;\
						  0.5 0 0.5;1 0 0.5;0 0.5 0.5;0.5 0.5 0.5;1 0.5 0.5;0 1 0.5;0.5 1 0.5;1 1 0.5;\
						  0 0 1;0.5 0 1;1 0 1;0 0.5 1;0.5 0.5 1;1 0.5 1;0 1 1;0.5 1 1;1 1 1]')
	ref_ind4e = np.matrix('[0 1 3 4 9 10 12 13;1 2 4 5 10 11 13 14;3 4 6 7 12 13 15 16;4 5 7 8 13 14 16 17;\
							9 10 12 13 18 19 21 22;10 11 13 14 19 20 22 23;12 13 15 16 21 22 24 25;13 14 16 17 22 23 25 26]')
	ref_n4e = np.matrix('[0 1 4 3 9 10 13 12;1 2 5 4 10 11 14 13;3 4 7 6 12 13 16 15;4 5 8 7 13 14 17 16;\
						  9 10 13 12 18 19 22 21;10 11 14 13 19 20 23 22;12 13 16 15 21 22 25 24;13 14 17 16 22 23 26 25]')
	ref_n4Db = np.squeeze(np.asarray(np.matrix('[0 1 2 3 4 5 6 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26]')))
	npt.assert_almost_equal(c4n, ref_c4n, decimal=8)
	npt.assert_almost_equal(ind4e, ref_ind4e , decimal=8)
	npt.assert_almost_equal(n4e, ref_n4e, decimal=8)
	npt.assert_almost_equal(n4Db, ref_n4Db, decimal=8)

def test_3d_uniform_cube_degree_two():
	from mozart.mesh.cube import cube
	x1, x2, y1, y2, z1, z2, Mx, My, Mz, N = (0, 1, 0, 1, 0, 1, 2, 2, 2, 2)
	c4n, ind4e, n4e, n4Db = cube(x1,x2,y1,y2,z1,z2,Mx,My,Mz,N)
	ref_c4n = np.matrix('[0 0 0;0.25 0 0;0.5 0 0;0.75 0 0;1 0 0;0 0.25 0;0.25 0.25 0;0.5 0.25 0;0.75 0.25 0;1 0.25 0;\
						  0 0.5 0;0.25 0.5 0;0.5 0.5 0;0.75 0.5 0;1 0.5 0;0 0.75 0;0.25 0.75 0;0.5 0.75 0;0.75 0.75 0;\
						  1 0.75 0;0 1 0;0.25 1 0;0.5 1 0;0.75 1 0;1 1 0;0 0 0.25;0.25 0 0.25;0.5 0 0.25;0.75 0 0.25;\
						  1 0 0.25;0 0.25 0.25;0.25 0.25 0.25;0.5 0.25 0.25;0.75 0.25 0.25;1 0.25 0.25;0 0.5 0.25;\
						  0.25 0.5 0.25;0.5 0.5 0.25;0.75 0.5 0.25;1 0.5 0.25;0 0.75 0.25;0.25 0.75 0.25;0.5 0.75 0.25;\
						  0.75 0.75 0.25;1 0.75 0.25;0 1 0.25;0.25 1 0.25;0.5 1 0.25;0.75 1 0.25;1 1 0.25;0 0 0.5;0.25 0 0.5;\
						  0.5 0 0.5;0.75 0 0.5;1 0 0.5;0 0.25 0.5;0.25 0.25 0.5;0.5 0.25 0.5;0.75 0.25 0.5;1 0.25 0.5;\
						  0 0.5 0.5;0.25 0.5 0.5;0.5 0.5 0.5;0.75 0.5 0.5;1 0.5 0.5;0 0.75 0.5;0.25 0.75 0.5;0.5 0.75 0.5;\
						  0.75 0.75 0.5;1 0.75 0.5;0 1 0.5;0.25 1 0.5;0.5 1 0.5;0.75 1 0.5;1 1 0.5;0 0 0.75;0.25 0 0.75;\
						  0.5 0 0.75;0.75 0 0.75;1 0 0.75;0 0.25 0.75;0.25 0.25 0.75;0.5 0.25 0.75;0.75 0.25 0.75;1 0.25 0.75;\
						  0 0.5 0.75;0.25 0.5 0.75;0.5 0.5 0.75;0.75 0.5 0.75;1 0.5 0.75;0 0.75 0.75;0.25 0.75 0.75;0.5 0.75 0.75;\
						  0.75 0.75 0.75;1 0.75 0.75;0 1 0.75;0.25 1 0.75;0.5 1 0.75;0.75 1 0.75;1 1 0.75;0 0 1;0.25 0 1;\
						  0.5 0 1;0.75 0 1;1 0 1;0 0.25 1;0.25 0.25 1;0.5 0.25 1;0.75 0.25 1;1 0.25 1;0 0.5 1;0.25 0.5 1;\
						  0.5 0.5 1;0.75 0.5 1;1 0.5 1;0 0.75 1;0.25 0.75 1;0.5 0.75 1;0.75 0.75 1;1 0.75 1;0 1 1;0.25 1 1;0.5 1 1;0.75 1 1;1 1 1]')
	ref_ind4e = np.matrix('[0 1 2 5 6 7 10 11 12 25 26 27 30 31 32 35 36 37 50 51 52 55 56 57 60 61 62;\
							2 3 4 7 8 9 12 13 14 27 28 29 32 33 34 37 38 39 52 53 54 57 58 59 62 63 64;\
							10 11 12 15 16 17 20 21 22 35 36 37 40 41 42 45 46 47 60 61 62 65 66 67 70 71 72;\
							12 13 14 17 18 19 22 23 24 37 38 39 42 43 44 47 48 49 62 63 64 67 68 69 72 73 74;\
							50 51 52 55 56 57 60 61 62 75 76 77 80 81 82 85 86 87 100 101 102 105 106 107 110 111 112;\
							52 53 54 57 58 59 62 63 64 77 78 79 82 83 84 87 88 89 102 103 104 107 108 109 112 113 114;\
							60 61 62 65 66 67 70 71 72 85 86 87 90 91 92 95 96 97 110 111 112 115 116 117 120 121 122;\
							62 63 64 67 68 69 72 73 74 87 88 89 92 93 94 97 98 99 112 113 114 117 118 119 122 123 124]')
	ref_n4e = np.matrix('[0 2 12 10 50 52 62 60;2 4 14 12 52 54 64 62;10 12 22 20 60 62 72 70;12 14 24 22 62 64 74 72;\
						  50 52 62 60 100 102 112 110;52 54 64 62 102 104 114 112;60 62 72 70 110 112 122 120;62 64 74 72 112 114 124 122]')
	ref_n4Db = np.squeeze(np.asarray(np.matrix('[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27\
												 28 29 30 34 35 39 40 44 45 46 47 48 49 50 51 52 53 54 55 59 60 64 65 69 70\
												 71 72 73 74 75 76 77 78 79 80 84 85 89 90 94 95 96 97 98 99 100 101 102 103\
												 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124]')))
	npt.assert_almost_equal(c4n, ref_c4n, decimal=8)
	npt.assert_almost_equal(ind4e, ref_ind4e, decimal=8)
	npt.assert_almost_equal(n4e, ref_n4e, decimal=8)
	npt.assert_almost_equal(n4Db, ref_n4Db, decimal=8)
