#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:27:19 2020

@author: fred
Miscellaneous utilities (may be split at some point):
    ++ Management of backends (custom for various users)
    ++ GPyOpt related functions
    
    
"""
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise.errors import ReadoutError
import qiskit as qk
import numpy as np
import os, socket, sys

NB_SHOTS_DEFAULT = 256
OPTIMIZATION_LEVEL_DEFAULT = 3
FULL_LIST_DEVICES = ['ibmq_poughkeepsie', 'ibmq_boeblingen', 'ibmq_singapore', 
             'ibmq_rochester', 'qasm_simulator']
# There may be more free devices
FREE_LIST_DEVICES = ['ibmq_16_melbourne', 'ibmq_vigo', 'qasm_simulator']


class BackendManager():
    """ Custom backend manager to deal with different users
    self.LIST_DEVICES : list of devices accessible to the user
    self.simulator: 'qasm_simulator' backend
    self.current_backend: 'current' backend by default the simulator but which 
        can be set updated by using get_backend method with inplace=True
    
    """
    def __init__(self):
        provider_free = qk.IBMQ.load_account()
        if 'kiran' in os.getcwd():
            self.LIST_OF_DEVICES = FULL_LIST_DEVICES
            provider_imperial = qk.IBMQ.get_provider(hub='ibmq', group='samsung', project='imperial')
            self.provider_list = {'free':provider_free, 'imperial':provider_imperial}
        else:
            self.LIST_OF_DEVICES = FREE_LIST_DEVICES
            self.provider_list = {'free':provider_free}
        self.simulator = qk.Aer.get_backend('qasm_simulator')
        self.current_backend = self.simulator

    # backend related utilities
    def print_backends(self):
        """List all providers by deafult or print your current provider"""
        #provider_list = {'Imperial':provider_free, 'Free':provider_imperial}
        for pro_k, pro_v in self.provider_list.items():
            print(pro_k)
            print('\n'.join(str(pro_v.backends()).split('IBMQBackend')))
            print('\n') 
        try:
            print('current backend:')
            print(self.current_backend.status())
        except:
            pass

    def get_backend(self, name, inplace=False):
        """ Gets back end preferencing the IMPERIAL provider
        Can pass in a named string or number corresponding to get_current_status output
        Comment: The name may be confusing as a method with the same name exists in qiskit
        """
        # check if simulator is chose
        if name == len(self.LIST_OF_DEVICES) or name == 'qasm_simulator':
            temp = self.simulator
        else: # otherwise look for number/name
            if type(name) == int: name = self.LIST_OF_DEVICES[name-1]
            try: #  tries imperial first
                temp = self.provider_list['imperial'].get_backend(name)
            except:
                temp = self.provider_list['free'].get_backend(name)
                
        # if inplace update the current backend
        if inplace:
            self.current_backend = temp
        return temp

    def get_current_status(self):
        """ Prints the status of each backend """
        for ct, device in enumerate(self.LIST_OF_DEVICES): # for each device
            ba = self.get_backend(device)
            print(ct+1, ':   ', ba.status()) # print status

    def gen_instance_from_current(self, nb_shots = NB_SHOTS_DEFAULT, 
                     optim_lvl = OPTIMIZATION_LEVEL_DEFAULT):
        """ Generate an instance from the current backend
        Not sure this is needed here: 
            + maybe building an instance should be decided in the main_script
            + maybe it should be done in the cost function
            + maybe there is no need for an instance and everything can be 
              dealt with transpile, compile
        """
        instance = qk.aqua.QuantumInstance(self.current_backend, shots=nb_shots,
                            optimization_level=optim_lvl)
        print('Generated a new quantum instance')
        return instance

    def gen_noise_model_from_backend(self, name_backend='ibmq_essex'):
        """ Given a backend name (or int) return the noise model associated"""
        backend = self.get_backend(name_backend)
        properties = backend.properties()
        noise_model = noise.device.basic_device_noise_model(properties)
        return noise_model


# BO related utilities
def add_path_GPyOpt():
    sys.path.insert(0, get_path_GPyOpt())
            
def get_path_GPyOpt():
    """ Generate the path where the package GPyOpt should be found, 
    this is custom to the user/machine
    it has been forked from https://github.com/FredericSauv/GPyOpt
    """
    if 'fred' in os.getcwd(): 
        path = '/home/fred/Desktop/GPyOpt/'
    elif 'level12' in socket.gethostname():
        path = '/home/kiran/QuantumOptimization/GPyOpt/'
    elif 'Lambda' in socket.gethostname():
        path = '/home/kiran/Documents/onedrive/Active_Research/QuantumSimulation/GPyOpt'
    else:
        path = ''
    return path

def get_best_from_bo(bo):
    """ Extract from a BO object the best set of parameters and fom
    based both from observed data and model"""
    x_obs = bo.x_opt
    y_obs = bo.fx_opt 
    pred = bo.model.predict(bo.X, with_noise=False)[0]
    x_pred = bo.X[np.argmin(pred)]
    y_pred = np.min(pred)
    return (x_obs, y_obs), (x_pred, y_pred)

def gen_res(bo):
    """ Generate a dictionary from a BO object to be stored"""
    (x_obs, y_obs), (x_pred, y_pred) = get_best_from_bo(bo)
    res = {'x_obs':x_obs, 
           'x_pred':x_pred, 
           'y_obs':y_obs, 
           'y_pred':y_pred,
           'X':bo.X, 
           'Y':bo.Y, 
           'gp_params':bo.model.model.param_array,
           'gp_params_names':bo.model.model.parameter_names()}
    return res

def gen_default_argsbo():
    """ maybe unnecessary"""
    default_args = {'initial_design_numdata':20,
           'model_update_interval':1, 
           'hp_update_interval':5, 
           'acquisition_type':'LCB', 
           'acquisition_weight':5, 
           'acquisition_weight_lindec':True, 
           'optim_num_anchor':5, 
           'optimize_restarts':1, 
           'optim_num_samples':10000, 
           'ARD':False}
    return default_args



# Generate noise models
def gen_ro_noisemodel(err_proba = [[0.1, 0.1],[0.1,0.1]], qubits=[0,1]): 
    noise_model = NoiseModel()
    for ro, q in zip(err_proba, qubits):
        err = [[1 - ro[0], ro[0]], [ro[1], 1 - ro[1]]]
        noise.add_readout_error(ReadoutError(err), [q])
    return noise_model


