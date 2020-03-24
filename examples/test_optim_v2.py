#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:10:36 2020

@author: fred
re-write test_GHZ with the new code
"""
import qiskit as qk
import numpy as np
import sys
sys.path.insert(0, '../core/')
import utilities as ut
import cost as cost
ut.add_path_GPyOpt()
import GPyOpt

NB_SHOTS_DEFAULT = 256
OPTIMIZATION_LEVEL_DEFAULT = 3
# ===================
# Choose a backend using the custom backend manager and generate an instance
# ===================
bem = ut.BackendManager()
bem.get_current_status()
chosen_device = int(input('SELECT IBM DEVICE:'))
bem.get_backend(chosen_device, inplace=True)
inst = bem.gen_instance_from_current(nb_shots=NB_SHOTS_DEFAULT, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)
inst_test = bem.gen_instance_from_current(nb_shots=8192, 
                                     optim_lvl=OPTIMIZATION_LEVEL_DEFAULT)

# ===================
# Define ansatz and initialize costfunction
# ===================
def ansatz(params):
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'), qk.QuantumRegister(1,'c'))
    c.rx(params[0], 0)
    c.rx(params[1], 1)
    c.ry(params[2], 2)
    c.barrier()
    c.cnot(0,2) 
    c.cnot(1,2) 
    c.barrier()
    c.rx(params[3], 0)
    c.rx(params[4], 1)
    c.ry(params[5], 2)
    c.barrier()
    return c

# different cost functions 
ghz_fidelity = cost.GHZPauliCost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
ghz_fidelity_test = cost.GHZPauliCost(ansatz=ansatz, instance = inst_test, N=3, nb_params=6)
ghz_f2 = cost.GHZWitness2Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
ghz_f1 = cost.GHZWitness1Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)

# ===================
# BO Optim
# No noise / Use of fidelity
# 20/25 works
#EPS = np.pi/2
#DOMAIN_RED = [(x-EPS, x+EPS) for x in X_SOL]
# ===================
# setup
NB_INIT = 30
NB_ITER = 20
X_SOL = np.pi/2 * np.array([[1.,1.,2.,1.,1.,1.]])
DOMAIN_FULL = [(0, 2*np.pi) for i in range(6)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
cost_bo = lambda x: 1-ghz_fidelity(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
ghz_fidelity_test(x_seen)
ghz_fidelity_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()


# ===================
# BO Optim
# No noise / Use of other cost function
# ===================
NB_INIT = 75
NB_ITER = 50
X_SOL = np.pi/2 * np.array([[1.,1.,2.,1.,1.,1.]])
DOMAIN_FULL = [(0, 2*np.pi) for i in range(6)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
cost_bo = lambda x: 1 - ghz_f2(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Look at results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
ghz_fidelity_test(x_seen)
ghz_fidelity_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# BO Optim
# No noise / Use of other cost function
# ===================
NB_INIT = 75
NB_ITER = 50
X_SOL = np.pi/2 * np.array([[1.,1.,2.,1.,1.,1.]])
DOMAIN_FULL = [(0, 2*np.pi) for i in range(6)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
cost_bo = lambda x: 1 - ghz_f1(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Look at results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
ghz_fidelity_test(x_seen)
ghz_fidelity_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()


## ===================
## Get a baseline to compare to BO
## ===================
#
#x_opt_guess =  np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
#x_opt_pred = Bopt.X[np.argmin(Bopt.model.predict(Bopt.X, with_noise=False)[0])]
#
#baseline_values = [qc.F(x_opt_guess) for ii in range(10)]
#bopt_values = [qc.F(x_opt_pred) for ii in range(10)]
#
#
#
#
#res_to_dill = gen_res(Bopt)
#dict_to_dill = {'Bopt_results':res_to_dill, 
#                'F_Baseline':baseline_values, 
#                'F_Bopt':bopt_values,
#                'Circ':qc.MAIN_CIRC[0],
#                'Device_config':qc.params_to_dict()}