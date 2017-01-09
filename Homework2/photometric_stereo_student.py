import math
import cv2
import numpy as np
import random

import scipy
from scipy import spatial



M = np.array([[     .1,     0,      0],
			  [      0,    .2,      0],
			  [      0,     0,     .3],
			  [.05774, .11547, .17321],
			  [.08165, .08165, .12247],
			  [.05345, .16036, .08018]])

def solve(M):

	U, Sigma, V = np.linalg.svd(M)

	SigmaMatrix = np.diag(Sigma)
	sqrtSig = np.sqrt(SigmaMatrix)

	a = np.zeros((3,3))
	sqrtSig = np.concatenate((sqrtSig, a))

	L_hat = np.dot(U,sqrtSig)
	S_hat = np.dot(sqrtSig,V)

	X_design = np.empty((6,6))
	for i in range(6):
		X_design[i,0] = L_hat[i,0]**2
		X_design[i,1] = 2*L_hat[i,0]*L_hat[i,1]
		X_design[i,2] = 2*L_hat[i,0]*L_hat[i,2]
		X_design[i,3] = L_hat[i,1]**2
		X_design[i,4] = 2*L_hat[i,1]*L_hat[i,2]
		X_design[i,5] = L_hat[i,2]**2

	ones = np.ones((6,1))
	B_part = np.linalg.lstsq(X_design,ones)
	B_part = B_part[0]
	B = np.empty((3,3))

	B[0,0] = B_part[0]
	B[0,1] = B_part[1]
	B[0,2] = B_part[2]
	B[1,0] = B_part[1]
	B[1,1] = B_part[3]
	B[1,2] = B_part[4]
	B[2,0] = B_part[2]
	B[2,1] = B_part[4]
	B[2,2] = B_part[5]

	u,s,v = np.linalg.svd(B)

	A = np.dot(u,np.sqrt(np.diag(s)))

	L_final = np.dot(L_hat,A)
	S_final = np.dot(np.linalg.inv(A),S_hat[:3,:])

	kd = np.zeros((3,1))
	for i in range(3):
		sum = 0
		sum += S_final[0,i]**2
		sum += S_final[1,i]**2
		sum += S_final[2,i]**2
		kd[i] = np.sqrt(sum)

	normals = S_final
	normals[:,0] /= kd[0]
	normals[:,1] /= kd[1]
	normals[:,2] /= kd[2]

	N = normals
	L = L_final
	S = S_final

	return {
	        'L_hat': L_hat,  # Pseudo-light matrix with dimensions (number of lights) x 3
	        'S_hat': S_hat,  # Pseudo-surface matrix with dimensions 3 x (number of points)
	        'X_design': X_design,  # The design matrix with dimensions 6 x 6 used to solve for B
	        'B': B,  # B with dimensions 3 x 3
	        'A': A,  # A with dimensions 3 x 3
	        'L': L,  # Solved light matrix with dimensions (number of lights) x 3
	        'S': S,  # Solved surface matrix with dimensions 3 x (number of points)
	        'kd': kd,  # Solved albedos vector with dimensions (number of points)
	        'N': N,  # Solved normals matrix with dimensions 3 x (number of points)
	    } 
