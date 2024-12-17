from utils import block_hankel
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import cvxpy as cp
from typing import Tuple

class npDeePC:

    """
    Vanilla regularized DeePC module
    """

    def __init__(self, ud: np.ndarray, yd: np.ndarray, y_constraints: Tuple[np.ndarray], u_constraints: Tuple[np.ndarray], 
                 N: int, Tini: int, n: int, p: int, m: int) -> None:
        
        """
        Initialise variables
        args:
            ud = Inpiut signal data
            yd = output signal data
            N = predicition horizon
            n = dimesnion of system
            p = output signal dimension
            m = input signal dimension
        """

        self.T = ud.shape[0]
        self.Tini = Tini
        self.n = n 
        self.N = N
        self.p = p
        self.m = m
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]

        # Check for full row rank
        H = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N+n, d=m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N, d=m)
        U_u, S_u, Vt_u = np.linalg.svd(U)
        #print(S_u)
        Y = block_hankel(w=yd.reshape((p*self.T,)), L=Tini+N, d=p)
        U_y, S_y, Vt_y = np.linalg.svd(Y)
        #print(S_y)
                
        self.Up = U[0:m*Tini,:]
        self.Yp = Y[0:p*Tini,:]
        self.Uf = U[Tini*m:,:]
        self.Yf = Y[Tini*p:,:]

        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(self.N*self.p)
        self.sig_y = cp.Variable(self.Tini*self.p)

        self.u_ini = cp.Parameter(self.Tini*self.m)
        self.y_ini = cp.Parameter(self.Tini*self.p)

        # Regularization Variables
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        self.PI = I - PI
        
    
    def setup(self, lam_g1=None, lam_g2=None, lam_y=None) -> None:
       
        """
        Set up controller constraints and cost function.
        Also used online during sim to update u_ini, y_ini, reference and regularizers
        args:
            ref = reference signal
            u_ini = initial input trajectory
            y_ini = initial output trajectory
            lam_g1, lam_g2 = regularization params for nonlinear systems
            lam_y = regularization params for stochastic systems
        """

        self.lam_y = lam_y
        self.lam_g1 = lam_g1
        self.lam_g2 = lam_g2
        #self.Q = np.kron(np.eye(self.N), Q)
        #self.R = np.kron(np.eye(self.N), R)
        

        Extender = np.kron(np.eye(self.N), np.ones((self.p, 1)))
        self.cost = cp.sum_squares(self.y-Extender@self.u)
        # #print(self.cost)
        #self.cost = cp.norm(self.u)
        #self.cost = cp.sum_squares(self.y - np.tile(self.u, self.p))
        if self.lam_y != None:
            self.cost += cp.norm(self.sig_y, 1)*self.lam_y
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini + self.sig_y,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower,
                cp.sum(self.g) == 1
            ]
        else:
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower,
                cp.sum(self.g) == 1
            ]

        if self.lam_g1 != None:
            self.cost += cp.sum_squares(self.PI@self.g)*lam_g1 
        if self.lam_g2 != None:
            self.cost += cp.norm(self.g, 1)*lam_g2
        assert self.cost.is_dpp

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self

    def solve(self, u_ini, y_ini, verbose=False, solver=cp.MOSEK, y_ref=0, u_ref=0) -> np.ndarray:
        
        """
        Call once the controller is set up with relevenat parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, usually use MOSEK
            verbose = bool for printing status of solver
        """

        # prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        # assert prob.is_dpp()
        # # assert prob.is_dcp()
        #self.y_ref.value = y_ref
        #self.u_ref.value = u_ref
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)

        #print(self.problem.variables()[1])
        #print(self.problem.variables()[1].value[:self.m])


        action = self.problem.variables()[0].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs
    
class npMPC:

    def __init__(self, A: np.ndarray, B: np.ndarray, Lambda: np.ndarray,
                 N: int, u_constraints: np.ndarray, y_constraints: np.ndarray) -> None:
        
        self.N = N
        self.p = B.shape[0] 
        self.N = N
        self.m = B.shape[1]
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]
        self.A = cp.Parameter(A.shape)
        self.B = cp.Parameter(B.shape)
        self.Lambda = cp.Parameter(Lambda.shape)
        #self.Q = Q
        #self.R = R
        self.A.value = A
        self.B.value = B
        self.Lambda.value = Lambda

        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.y = cp.Variable(self.N*self.p)
        self.y_ref = cp.Parameter((self.N*self.p,))
        self.u_ref = cp.Parameter((self.N*self.m,))
        self.y_ini = cp.Parameter(self.p)

    def setup(self, x0 = np.ndarray):

        self.x0 = cp.Parameter(x0.shape)
        self.x0.value = x0
        #self.Q = np.kron(np.eye(self.N), self.Q)
        #self.R = np.kron(np.eye(self.N), self.R)
        #self.cost = cp.quad_form(self.y-self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u-self.u_ref, cp.psd_wrap(self.R))
        Extender = np.kron(np.eye(self.N), np.ones((self.p, 1)))
        self.cost = cp.sum_squares(self.y-Extender@self.u)
        
        #self.cost = cp.quad_form(self.u, np.eye(self.N*self.m))

        self.constraints = [
            self.y[:self.p] == self.y_ini,
            self.u <= self.u_upper, self.u >= self.u_lower,
            self.y <= self.y_upper, self.y >= self.y_lower
        ]

        for i in range(1,self.N):
            self.constraints.append(
                self.y[self.p*i:self.p*(i+1)] == self.A@self.y[self.p*(i-1):self.p*i] + self.B@self.u[self.m*(i-1):self.m*i] + self.Lambda@self.y[self.p*(i-1):self.p*i]
            )

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self
    
    def solve(self, y_ini, u_ini=None, verbose=False, solver=cp.OSQP) -> np.ndarray:
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)
        action = self.problem.variables()[1].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs