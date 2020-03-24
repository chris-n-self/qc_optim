#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:24:40 2020

@author: form Kiran
"""

import qiskit as qk
import numpy as np
import os, time




MEAS_DEFAULT = ['xxx', '1zz', 'z1z', 'zz1', 'yyx', 'xyy', 'yxy']
MEAS_WEIGHTS = np.array([1., 1., 1., 1., -1., -1., -1.])/4.0
NB_SHOTS_DEFAULT = 256
OPTIMIZATION_LEVEL_DEFAULT = 3
LIST_OF_DEVICES = ['ibmq_poughkeepsie', 'ibmq_boeblingen', 'ibmq_singapore', 'ibmq_rochester', 'qasm_simulator']
META_DATA = []
VARIABLE_PARAMS = [qk.circuit.Parameter('R1'),
                    qk.circuit.Parameter('R2'),
                    qk.circuit.Parameter('R3'),
                    qk.circuit.Parameter('R4'),
                    qk.circuit.Parameter('R5'),
                    qk.circuit.Parameter('R6')]

pi = np.pi






# Sorry about this cluster fuck - hacked way to run smoothly with or without imperials token
try:
    provider_free
except:
    provider_free = qk.IBMQ.load_account()
    simulator = qk.Aer.get_backend('qasm_simulator')

    if 'kiran' in os.getcwd():
        provider_imperial = qk.IBMQ.get_provider(hub='ibmq', group='samsung', project='imperial')
        provider_list = {'free':provider_free, 'imperial':provider_imperial}
    else:
        provider_list = {'free':provider_free}
        LIST_OF_DEVICES = ['ibmq_16_melbourne', 'ibmq_vigo', 'qasm_simulator']
    





# Choosing this a the defaul simulator
actual_backend = simulator
instance = qk.aqua.QuantumInstance(actual_backend, 
                                   shots=NB_SHOTS_DEFAULT, 
                                   optimization_level=OPTIMIZATION_LEVEL_DEFAULT)




# %% 

# ------------------------------------------------------
# Deals with circuit creation, measurement and param bindings - move some to core? 
# ------------------------------------------------------




def create_circ(params = VARIABLE_PARAMS):
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




def append_measurements(circ, measurements):
    """ Assumes creator returns an instance of the relevant circuit"""
    num_classical = len(measurements.replace('1',''))
    if num_classical > 0:
        cr = qk.ClassicalRegister(num_classical, 'classical')
        circ.add_register(cr)
    ct_m = 0
    ct_q = 0
    for basis in measurements:
        if basis == 'z':
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'x':
            circ.u3(pi/2, 0, 0, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'y':
            circ.u3(pi/2, -pi/2, -pi/2, ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == '1':
            pass
        ct_q+=1
    return circ





def gen_meas_circuits(creator = create_circ, 
                      meas_settings = MEAS_DEFAULT, 
                      params = VARIABLE_PARAMS):
    """ Return a list of measurable circuit with same parameters but with 
    different measurement settings"""
    c_list = [append_measurements(creator(params), m)for m in meas_settings]
    return c_list




# might need updateing to generalize
def update_params(circ, val_dict=np.zeros(6)):
    '''Returns list of circuit with bound values DOES NOT MODIFY INPUT'''
    if type(circ) != list: 
        circ = [circ]
    if type (val_dict) is not dict:
        val_dict = {key:val for key,val in zip(VARIABLE_PARAMS,val_dict)}
    bound_circ = []
    for cc in circ:
        bound_circ.append(cc.bind_parameters(val_dict))
    return bound_circ  

# %%
# ------------------------------------------------------
# Im not happy about this implimentation 
# ------------------------------------------------------
MAIN_CIRC = instance.transpile(gen_meas_circuits())

def UpdateQuantumCircuit(creator = create_circ, 
                         meas_settings = MEAS_DEFAULT, 
                         params = VARIABLE_PARAMS):
    global MAIN_CIRC
    print('Warning this function may reset which qubits are used')
    MAIN_CIRC = instance.transpile(gen_meas_circuits(creator = create_circ, 
                                  meas_settings = MEAS_DEFAULT, 
                                  params = VARIABLE_PARAMS)
                                   )




# %% 
# ------------------------------------------------------
# Will eventually be mostly moved over to core? 
# ------------------------------------------------------

def freq_even(results):
    """ Frequency of +1 eigen values: result 0(1) corresponds to a -1(+1) eigen 
    state. If even number of 0 then +1
    """
    nb_odd, nb_even = 0, 0
    for k, v in results.items():
        nb_even += v * (k.count('0')%2 == 0)
        nb_odd += v * (k.count('0')%2)
    return nb_even / (nb_odd + nb_even)



def F(experimental_params, 
      flags = False, 
      meas_weights=MEAS_WEIGHTS, 
      shots = NB_SHOTS_DEFAULT,
      noise_model = None):
    """ Main function: take parameters and return an estimation of the fidelity 
    Different possible behaviors:
        +
        + Shots is now handeled in the ``instance'' object, (needed to force every circuit to be identical)
        + Handeling of all circuit params is now passed to UpdateQuantumCircuit 
            + making use of 'quantum_instance' 
            + Handels circuits are on the same qubits and communication errors
            + removed noise model having any effect - will add back in 

    """
    # if type(meas_settings) is not list: meas_settings = [meas_settings]
    # if type(shots) == int: shots = [shots] * len(meas_settings) number of shots is not defined in the ``instance''
    circs = update_params(MAIN_CIRC, experimental_params)
    

################
    # update here to include the noise model
    results = instance.execute(circs, had_transpiled=True)
################
    
    counts = [results.get_counts(ii) for ii in range(len(circs))]
    
    measurement_results = [freq_even(ct) for ct in counts]
    
    META_DATA.append(results.to_dict()) #keep data
    
    if meas_weights is None:
        res = measurement_results
    else:
        res = np.dot(measurement_results, meas_weights)
    return np.squeeze(res) #, meta_results


# %%
# ------------------------------------------------------
# Deals with looking at backends / chosing a new device
# ------------------------------------------------------

    
def ListBackends():
    '''List all providers by deafult or print your current provider'''
    #provider_list = {'Imperial':provider_free, 'Free':provider_imperial}
    for pro in list(provider_list.keys()):
        print(pro)
        pro = provider_list[pro]
        print('\n'.join(str(pro.backends()).split('IBMQBackend')))
        print('\n') 
    try:
        print('current backend:')
        print(actual_backend.status())
    except:
        pass

    
def GetBackend(name, inplace=False):
    '''Gets back end preferencing the IMPERIAL provider
    Can pass in a named string or number from CurrentStatus output'''
    global actual_backend, instance
    
    if name == len(LIST_OF_DEVICES) or name == 'qasm_simulator':      # check if simulator is chosen
        temp = simulator
    else: # otherwise look for number/name
        if type(name) == int: name = LIST_OF_DEVICES[name-1]
        try: #  tries imperial first
            temp = provider_imperial.get_backend(name)
        except:
            temp = provider_free.get_backend(name)
            
    # if inplace update the current environment variables
    if inplace:
        actual_backend = temp
        instance = qk.aqua.QuantumInstance(actual_backend, 
                                           shots=NB_SHOTS_DEFAULT, 
                                           optimization_level=OPTIMIZATION_LEVEL_DEFAULT)
        print('Warning also updated quantum instance to default values')
    return temp
        

def CurrentStatus():
    '''Prints the status of each backend'''
    ct = 0
    for device in LIST_OF_DEVICES: # for each device
        ba = GetBackend(device);ct+=1
        print(ct, ':   ', ba.status()) # print status
  


# %%
# ------------------------------------------------------
# This helps export relevant environment params
# ------------------------------------------------------

def params_to_dict():
    di = {'shots':NB_SHOTS_DEFAULT,
     'optimization_level':OPTIMIZATION_LEVEL_DEFAULT,
     'device':actual_backend.name(),
     'time_ns':time.time_ns()}
    return di

def reset_meta_data():
    '''Hard reset of the META_DATA variable'''
    global META_DATA
    del(META_DATA) # should probaby ask for user prompt
    META_DATA = []



# %% Test
if __name__ == '__main__':
    x_opt = np.array([3., 3., 2., 3., 3., 1.]) * np.pi/2
    x_loc = np.array([1., 0., 4., 0., 3., 0.]) * np.pi/2
    print(F(x_loc)) #~0.5
    print(F(x_opt)) # 1.0
    
