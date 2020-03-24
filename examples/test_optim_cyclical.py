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

NB_SHOTS_DEFAULT = 8192
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
# Todo: generalize to abitrary nb of qubits
# ===================
def ansatz_easy(params):         
    """ Ansatz for which an ideal solution exist"""
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'),
                          qk.QuantumRegister(1,'c'),qk.QuantumRegister(1,'d'),
                          qk.QuantumRegister(1,'e'),qk.QuantumRegister(1,'f'))
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    c.barrier()
    c.cu1(np.pi,0,1)
    c.cu1(np.pi,2,3)
    c.cu1(np.pi,4,5)
    c.barrier()
    c.cu1(np.pi,1,2)
    c.cu1(np.pi,3,4)
    c.cu1(np.pi,5,0)
    c.barrier()
    return c
    
def ansatz_hard(params):
    """ Ansatz to be refined"""
    c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(1, 'b'),
                          qk.QuantumRegister(1,'c'),qk.QuantumRegister(1,'d'),
                          qk.QuantumRegister(1,'e'),qk.QuantumRegister(1,'f'))
    c.ry(params[0],0)
    c.ry(params[1],1)
    c.ry(params[2],2)
    c.ry(params[3],3)
    c.ry(params[4],4)
    c.ry(params[5],5)
    c.barrier()
    c.cnot(0,1) 
    c.cnot(2,3) 
    c.cnot(4,5)
    c.cnot(1,2) 
    c.cnot(3,5)
    c.cnot(4,0)
    c.barrier()
    c.ry(params[6], 0)
    c.ry(params[7], 1)
    c.ry(params[8], 2)
    c.rx(params[9], 3)
    c.ry(params[10], 4)
    c.rx(params[11], 5)
    c.barrier()
    return c

### (ansatz, nb_params, nb_qubits, sol)
pb_infos = [(ansatz_easy, 6, 6, np.pi/2 * np.ones(shape=(6,))),
             (ansatz_hard, 12, 6, None)]
ansatz, nb_p, nb_q, x_sol = pb_infos[0]

# different cost functions 
fid = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
fid_test = cost.GraphCyclPauliCost(ansatz=ansatz, instance = inst_test, N=nb_q, nb_params=nb_p)
cost2 = cost.GraphCyclWitness1Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)
cost1 = cost.GraphCyclWitness2Cost(ansatz=ansatz, instance = inst, N=nb_q, nb_params=nb_p)


if x_sol is not None:
    assert fid_test(x_sol) == 1., "pb with ansatz/x_sol"
    assert cost1(x_sol) == 1., "pb with ansatz/x_sol"
    assert cost2(x_sol) == 1., "pb with ansatz/x_sol"

# ===================
# BO Optim: no noise / Use of fidelity
# 20/25 works
#EPS = np.pi/2
#DOMAIN_RED = [(x-EPS, x+EPS) for x in X_SOL]
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
cost_bo = lambda x: 1-fid(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

fid_test(x_sol)


# ===================
# BO Optim
# Cost function 1 (Best one so far)
# ===================
# setup
NB_INIT = 50
NB_ITER = 50
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
cost_bo = lambda x: 1-cost1(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
print(Bopt.model.model)
Bopt.plot_convergence()

# ===================
# BO Optim
# No noise / Cost Function 2 
# seems to work better with this split 30/70 not so much with 50/50 and more 
# exploration
# ===================
# setup
NB_INIT = 30
NB_ITER = 70
DOMAIN_FULL = [(0, 2*np.pi) for i in range(nb_p)]
DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
bo_args = ut.gen_default_argsbo()
bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT, 
                'acquisition_weight': 6})
cost_bo = lambda x: 1-cost2(x) 

#optim
Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
print("start optim")
Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

# Results found
(x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
fid_test(x_seen)
fid_test(x_exp)
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