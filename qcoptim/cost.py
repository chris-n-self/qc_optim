#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:11:28 2020

@author: fred
TODO: (IN PROGRESS)implement cyclical graph states 
TODO: (SOON) implement more general graph states 
TODO: (LATER) ability to deal with different number of shots 
TODO: (LATER) implement sampling (of the measurement settings) strat 
TODO: (LATER) more flexible number of shots
TODO: (LATER)how to make sure that two cost functions are computed based on the 
same transpiled circuits (e.g. fidelity and another cost function)

Choice of noise_models, initial_layouts, nb_shots, etc.. is done through the 
quantum instance passed when initializing a Cost, i.e. it is outside of the
scope of the classes here
"""
__all__ = [
    'CostWeightedOps',
]

import qiskit as qk
import numpy as np
import pdb
#import itertools as it
pi =np.pi

from qiskit.aqua.operators import WeightedPauliOperator, TPBGroupedWeightedPauliOperator


class CostWeightedOps(object):

    def __init__(self, 
                 ansatz, 
                 instance, 
                 operators,
                 fix_transpile=True,
                 keep_res=True, 
                 verbose=True, 
                 noise_model=None,
                 debug=False,
                 **kwargs,
                ):
        """ """
        if debug: pdb.set_trace()
        self.ansatz = ansatz
        self.instance = instance
        self.fix_transpile = fix_transpile
        self.verbose = verbose
        self._keep_res = keep_res
        self._res = []

        # check type of passed operators
        if not type(operators) is WeightedPauliOperator:
            raise TypeError
        # store operators in grouped form, currently use `unsorted_grouping` method, which
        # is a greedy method. Sorting method could be controlled with a kwarg
        self.grouped_weighted_operators = TPBGroupedWeightedPauliOperator.unsorted_grouping(operators)

        # generate and transpile measurement circuits
        measurement_circuits = self.grouped_weighted_operators.construct_evaluation_circuit(
            wave_function=self.ansatz.circuit,
            statevector_mode=self.instance.is_statevector)
        self._t_measurement_circuits = self.instance.transpile(measurement_circuits)
        self.num_circuits = len(self._t_measurement_circuits)

    def __call__(self, params_values, debug=False):
        """ Estimate the CostFunction for some parameters"""
        if debug: pdb.set_trace()

        if np.ndim(params_values)==1:
            params_values = [params_values]

        # package and bind circuits
        bound_circs = []
        for pidx,p in enumerate(params_values):
            for cc in self._t_measurement_circuits:
                tmp = cc.bind_parameters(dict(zip(self.ansatz.params, p)))
                tmp.name = str(pidx) + tmp.name
                bound_circs.append(tmp)
            
        # See if one can add a noise model here and the number of parameters
        results = self.instance.execute(bound_circs, had_transpiled=self.fix_transpile)
        if self._keep_res: self._res.append(results.to_dict())
            
        # evaluate mean value of sum of grouped_weighted_operators from results
        means = []
        for pidx,p in enumerate(params_values):
            mean,std = self.grouped_weighted_operators.evaluate_with_result(
                results,statevector_mode=self.instance.is_statevector,circuit_name_prefix=str(pidx))
            means.append(mean)

        if self.verbose: print(means)
        return np.array([np.squeeze(means)]).T


class Cost():
    """
    This base class defines all the ingredients necessary to evaluate a cost 
    function based on an ansatz circuit:
        + what should be measured and how many times
        + how the measurements outcomes(counts) should be aggregated to return 
          an estimate of the cost
        + how should be the full(ansatz+measurements) circuit generated and 
          transpiled
        + what results to keep 
    
    Logic of computing the cost are defined by the followings (which are not
    implement in the base class but should be implemented in the subclasses):
        + self._list_meas is a list of M strings indicating all the measurement 
            settings required
        + self._meas_func is a single function taking as an input the list of 
            the M outputs from the execution of the circuits (counts dictionaries) 
            and returning a single value
    
    Other bits of logic have been included:
        + transpile
          ++ if False the circuit is only transpiled once. Then 
             in this case 'args' should contain 
          ++if True the circuit is transpiled at each call of the function
        + keep_res 
          ++ if True the results from the execution of the circuits are kept 
             (appended) in self._res
                    
    Questions: 
        + should we pass backend or instance
        + Do we need to pass the number of shots in the instance when transpiling or can they 
        be specified at exec time
    append_measurements
    
    Terminology:
        + ansatz: callable function (may be a callable object in the future) 
                 which takes parameters as input and return a circuit
        + circuit: quantum circuit 
        + measurable circuit: circuit with measurement operations
        
    """

    def __init__(self, ansatz, N,  instance, nb_params, fix_transpile = True,
                  keep_res = True, verbose = True, noise_model = None,
                  debug=False, **args):
        """ initialize the cost function taking as input parameters:
            + ansatz 
            + N <int>: the number of qubits
            + simulator
        """
        if debug: pdb.set_trace()
        self.ansatz = ansatz
        self.instance = instance
        self.nb_qubits = N  # may be redundant
        self.dim = np.power(2,N)
        self.nb_params = nb_params # maybe redundant
        self.fix_transpile = fix_transpile
        self.verbose = verbose
        self._keep_res = keep_res
        self._res = []

        # These methods needs to be implemented in the subclasses
        self._list_meas = self._gen_list_meas()  
        self._meas_func = self._gen_meas_func() 

        # measurable circuits are transpiled at initialization
        self._gen_qk_vars()
        self._transpile_measurable_circuits(self._qk_vars)
    
    def __call__(self, params, debug=False):
        """ Estimate the CostFunction for some parameters"""
        if debug: pdb.set_trace()
        if np.ndim(params) > 1 :
            res = np.array([self.__call__(p) for p in params])
        else:
            #bind the values of the parameters to the circuit
            bound_circs = bind_params(self._main_circuit, params, self._qk_vars)
            
            #See if one can add a noise model here and the number of parameters
            results = self.instance.execute(bound_circs, had_transpiled=self.fix_transpile)
            
            if self._keep_res: self._res.append(results.to_dict())
            counts = [results.get_counts(ii) for ii in range(len(self._list_meas))]
            res = self._meas_func(counts)    
            if self.verbose: print(res)
        return np.squeeze(res) 
    
    def _transpile_measurable_circuits(self, params):
        """ Transpile all the measurable circuits"""
        instance = self.instance
        ansatz = self.ansatz
        meas_settings = self._list_meas
        list_circ_meas = gen_meas_circuits(ansatz, meas_settings, params)
        self._main_circuit = instance.transpile(list_circ_meas)
        print('Measuremable circuits have been transpiled')

    def _gen_qk_vars(self):
        """ Generate qiskit variables to be bound to a circuit"""
        name_params = ['R'+str(i) for i in range(self.nb_params)]
        self._qk_vars = [qk.circuit.Parameter(n) for n in name_params]
        
    def _init_res(self):
        """ Flush the res accumulated so far """
        self._res = []

    def _gen_list_meas(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()
        
    def _gen_meas_func(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError()


# Subclasses: GHZ related costs
class GHZPauliCost(Cost):
    """ Cost = fidelity w.r.t. a N-qubit GHZ state, estimated based on the 
    expected values of N-fold Pauli operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for GHZ of different sizes
    # {'nb_qubits':(meas_strings, weights)}, wehere the measurement string is a 
    # list of Pauli operators and the weights correspond to the decomposition 
    # of the GHZ state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _GHZ_PAULI_DECOMP = {
    '2':(
            ['xx', 'yy', 'zz'], 
            np.array([1.,-1.,1.])
            ),
    '3':(
            ['1zz','xxx','xyy','yxy','yyx','z1z','zz1'], 
            np.array([1., 1., -1., -1., -1., 1.,1.])
            ),
    '4':( 
            ['11zz','1z1z','1zz1','xxxx','xxyy','xyxy','xyyx','yxxy','yxyx',
             'yyxx','yyyy','z11z','z1z1','zz11','zzzz'],
            np.array([1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,1.,1.,1.])
            )
        }
        
    def _gen_list_meas(self):
        return self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._GHZ_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim

        return meas_func


class GHZWitness1Cost(Cost):
    """ Cost based on witnesses for genuine entanglement ([guhne2005])
    Stabilizer generators S_l of GHZ are (for n=4) S = <XXXX, ZZII, IZZI, IIZZ>
    To estimate S_1 to S_n only requires two measurement settings: XXXX, ZZZZ
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return 0.5*(S1-1) + np.prod((S2+1)/2)
        return meas_func

class GHZWitness2Cost(Cost):
    """ Exactly as GHZWitness1Cost except that Cost =  Sum_l[S_l] - (N-1)I """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['x...x', 'z...z']"""
        N = self.nb_qubits
        list_meas = ['x'*N, 'z'*N]
        return list_meas
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        def meas_func(counts):
            S1 = freq_even(counts[0])
            S2 = np.array([freq_even(counts[1], indices=[i,i+1]) for i in range(N-1)])
            return S1 + np.sum(S2) - (N -1)
        return meas_func
    
    
# Subclasses: Graph states
class GraphCyclPauliCost(Cost):
    """ A N-qubit Cyclical graph has edges = [[1,2],[2,3],...,[N-1,N],[N,1]]
    Cost = fidelity, estimated based on the expected values of the N-fold Pauli 
    operators (e.g. 'XXY')
    """   
    # Hardcoded list of measurements settings for Cyclical graph states of 
    #different sizes {'nb_qubits':(meas_strings, weights)}, wehere the measurement 
    # string is a list of Pauli operators and the weights correspond to the 
    # decomposition of the target state in the Pauli tensor basis (up to a constant 1/dim)
    # It could be automated to deal with arbitrary size state
    _CYCLICAL_PAULI_DECOMP = {
    '2':(
            ['1x','x1','xx'], 
            np.array([1,1,1])
            ),
    '3':(
            ['1yy','xxx','xzz','y1y','yy1','zxz','zzx'], 
            np.array([1,-1,1,1,1,1,1])
            ),
    '4':( 
            ['1x1x','1yxy','1zxz','x1x1','xxxx','xy1y','xz1z','y1yx','yxy1','yyzz',
             'yzzy','z1zx','zxz1','zyyz','zzyy'],
            np.array([1,-1,1,1,1,-1,1,-1,-1,1,1,1,1,1,1])
            ),
    '5':( 
            ['11zxz','1x1yy','1xzzx','1yxxy','1yy1x','1zxz1','1zyyz','x1xzz','x1yy1',
             'xxxxx','xxy1y','xy1yx','xyzzy','xz11z','xzzx1','y1x1y','y1yxx','yxxy1',
             'yxyzz','yy1x1','yyz1z','yz1zy','yzzyx','z11zx','z1zyy','zx1xz','zxz11',
             'zyxyz','zyyz1','zzx1x','zzyxy'],
            np.array([1,1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1])
            ),
    '6':( 
            ['111zxz','11zxz1','11zyyz','1x1x1x','1x1yxy','1xz1zx','1xzzyy','1yxxxy',
             '1yxy1x','1yy1yy','1yyzzx','1zx1xz','1zxz11','1zyxyz','1zyyz1','x1x1x1',
             'x1xz1z','x1yxy1','x1yyzz','xxxxxx','xxxy1y','xxy1yx','xxyzzy','xy1x1y',
             'xy1yxx','xyz1zy','xyzzyx','xz111z','xz1zx1','xzzxzz','xzzyy1','y1x1yx',
             'y1xzzy','y1yxxx','y1yy1y','yxxxy1','yxxyzz','yxy1x1','yxyz1z','yy1xzz',
             'yy1yy1','yyz11z','yyzzx1','yz11zy','yz1zyx','yzzx1y','yzzyxx','z111zx',
             'z11zyy','z1zx1x','z1zyxy','zx1xz1','zx1yyz','zxz111','zxzzxz','zyxxyz',
             'zyxyz1','zyy1xz','zyyz11','zzx1yy','zzxzzx','zzyxxy','zzyy1x'],
            np.array([1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,
                      -1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,1,1,1,
                      -1,1,1,1,1,1,-1,1,1,1,1,1,1])
            )
        }
        
    def _gen_list_meas(self):
        return self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][0]
    
    def _gen_meas_func(self):
        """ expected parity associated to each of the measurement settings"""
        weights = self._CYCLICAL_PAULI_DECOMP[str(self.nb_qubits)][1]
        dim = self.dim
        def meas_func(counts):
            return (1+np.dot([expected_parity(c) for c in counts], weights))/dim

        return meas_func


class GraphCyclWitness1Cost(Cost):
    """ Cost based on witnesses for genuine entanglement ([guhne2005])
    Stabilizer generators S_l of cyclical graph states are (for N=4 qubits) 
        S = <XZIZ, ZXZI, IZXZ, ZIZX>
    To estimate S_1 to S_N only requires two measurement settings: XZXZ, ZXZX
    Cost =  (S_1 - 1)/2 + Prod_l>1 [(S_l + 1)/2] 
    !!! ONLY WORK FOR EVEN N FOR NOW !!!
    !!! MORE EXPLAIN NEEDED !!!
    """   
    
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas_odd = "".join(['zx'] * (N//2))
            meas_even = "".join(['xz'] * (N//2))
        return [meas_odd, meas_even]
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return 0.5*(S_even[-1]-1) + np.prod((S_odd+1)/2) * np.prod((S_even[:-1]+1)/2) 
        return meas_func

class GraphCyclWitness2Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that Cost =  Sum_l[S_l] - (N-1)I """   
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            meas1 = "".join(['zx'] * (N//2))
            meas2 = "".join(['xz'] * (N//2))
        return [meas1, meas2]
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        N = self.nb_qubits
        if (N%2): 
            raise NotImplementedError("ATM cannot deal with odd N")
        else:
            ind_odd = [[i, i+1, i+2] for i in range(0,N-2, 2)] + [[0, N-2, N-1]]  
            ind_even = [[i, i+1, i+2] for i in range(1,N-2, 2)] + [[0, 1, N-1]]
            def meas_func(counts):
                counts_odd, counts_even = counts[0], counts[1]
                S_odd = np.array([expected_parity(counts_odd, indices=i) for i in ind_odd])
                S_even = np.array([expected_parity(counts_even, indices=i) for i in ind_even])
                return np.sum(S_odd) + np.sum(S_even) - (N-1)
        return meas_func

class GraphCyclWitness3Cost(Cost):
    """ Exactly as GraphCyclWitness1Cost except that Cost =  XXX
    To implement"""   
    def _gen_list_meas(self):
        """ two measurement settings ['zxzx...zxz', 'xzxzx...xzx']"""
        raise NotImplementedError("To be implemented")
    
    def _gen_meas_func(self):
        """ functions defining how outcome counts should be used """
        raise NotImplementedError("To be implemented")
        
# ------------------------------------------------------
# Functions to compute expected values based on measurement outcomes counts as 
# returned by qiskit
# ------------------------------------------------------
def freq_even(count_result, indices=None):
    """ return the frequency of +1 eigenvalues:
    The +1 e.v. case corresponds to the case where the number of 0 in the 
    outcome string is even
    
    indices: list<integer>
             if not None it allows to consider only selected elements of the 
             outcome string
    """
    nb_odd, nb_even = 0, 0
    for k, v in count_result.items():
        k_invert = k[::-1]
        sub_k = get_substring(k_invert, indices)
        nb_even += v * (sub_k.count('1')%2 == 0)
        nb_odd += v * (sub_k.count('1')%2)
    return nb_even / (nb_odd + nb_even)

def expected_parity(results,indices=None):
    """ return the estimated value of the expectation of the parity operator:
    P = P+ - P- where P+(-) is the projector 
    Comment: Parity operator may nor be the right name
    """
    return 2 * freq_even(results, indices=indices) - 1


def get_substring(string, list_indices=None):
    """ return a substring comprised of only the elements associated to the 
    list of indices
    Comment: probably already exist or there may be a better way"""
    if list_indices == None:
        return string
    else:
        return "".join([string[ind] for ind in list_indices])

# ------------------------------------------------------
# Some functions to deals with appending measurement and param bindings  
# ------------------------------------------------------
def append_measurements(circ, measurements):
    """ Assumes circ returns an instance of the relevant circuit"""
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
            #circ.u3(pi/2, 0, 0, ct_q)
            circ.h(ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == 'y':
            #circ.u3(pi/2, -pi/2, -pi/2, ct_q)
            circ.sdg(ct_q)
            circ.h(ct_q)
            circ.measure(ct_q, ct_m)
            ct_m+=1
        elif basis == '1':
            pass
        ct_q+=1
    return circ


def gen_meas_circuits(ansatz, meas_settings, params):
    """ Return a list of measurable circuit with same parameters but with 
    different measurement settings"""
    c_list = [append_measurements(ansatz(params), m) for m in meas_settings]
    return c_list


def bind_params(circ, param_values, param_variables):
    """ Take a list of circuits with bindable parameters and bind the values 
    passed according to the param_variables
    Returns the list of circuits with bound values DOES NOT MODIFY INPUT
    (i.e. hardware details??)
    """
    if type(circ) != list: 
        circ = [circ]
    val_dict = {key:val for key,val in zip(param_variables, param_values)}
    bound_circ = [cc.bind_parameters(val_dict) for cc in circ]
    return bound_circ  




if __name__ == '__main__':
    #-----#
    # Verif conventions
    #-----#
    def ansatz(params):
        c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), qk.QuantumRegister(2, 'b'))
        c.h(0)
        return c
    
    sim = qk.Aer.get_backend('qasm_simulator')
    inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
    m_c = gen_meas_circuits(ansatz, ['zz'], [])
    res = inst.execute(m_c)
    counts = res.get_counts()
    
    #-----#
    # GHZ
    #-----#
    # Create an ansatz capable of generating a GHZ state (not the most obvious 
    # one here) with the set of params X_SOL
    def ansatz(params):
        c = qk.QuantumCircuit(qk.QuantumRegister(1, 'a'), 
                        qk.QuantumRegister(1, 'b'), qk.QuantumRegister(1,'c'))
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
    X_SOL = np.pi/2 * np.array([1.,1.,2.,1.,1.,1.])
    X_LOC = np.pi/2 * np.array([1., 0., 4., 0., 3., 0.])
    X_RDM = np.random.uniform(0.0, 2*pi, size=(6,1))
    
    # Create an instance
    sim = qk.Aer.get_backend('qasm_simulator')
    inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
    
    # Verif the values of the different GHZ cost
    # Fidelity
    ghz_cost = GHZPauliCost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
    assert ghz_cost(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
    assert np.abs(ghz_cost(X_LOC) - 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.5 (up to sampling error)"
    
    # Witnesses inspired cost functions: they are different compared to the fidelity
    # but get maximized only when the state is the right one
    ghz_witness1 = GHZWitness1Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
    assert ghz_witness1(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
    assert np.abs(ghz_witness1(X_LOC) - 0.31) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"

    ghz_witness2 = GHZWitness2Cost(ansatz=ansatz, instance = inst, N=3, nb_params=6)
    assert ghz_witness2(X_SOL) == 1.0, "For this ansatz, parameters, cost function should be one"
    assert np.abs(ghz_witness2(X_LOC) + 0.5) < 0.1, "For this ansatz and parameters, the cost function should be close to 0.31 (up to sampling error)"    
    
    
    
    #-----#
    # Cyclical graph states
    #-----#
    N_graph = 6
    N_params = 6
    def ansatz(params):
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
        c.cu1(pi,0,1)
        c.cu1(pi,2,3)
        c.cu1(pi,4,5)
        c.barrier()
        c.cu1(pi,1,2)
        c.cu1(pi,3,4)
        c.cu1(pi,5,0)
        c.barrier()
        return c

    X_SOL = np.pi/2 * np.ones(N_params) # sol of the cycl graph state for this ansatz
    X_RDM = np.array([1.70386471,1.38266762,3.4257722,5.78064,3.84102323,2.37653078])
    #X_RDM = np.random.uniform(low=0., high=2*np.pi, size=(N_params,))
    
    # Create an instance
    sim = qk.Aer.get_backend('qasm_simulator')
    inst = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)
    graph_cost = GraphCyclPauliCost(ansatz=ansatz, instance = inst, N=N_graph
                                    , nb_params=N_params)
    fid_opt = graph_cost(X_SOL)
    fid_rdm = graph_cost(X_RDM)
    assert fid_opt == 1.0, "For this ansatz, parameters, cost function should be one"
    assert (fid_opt-fid_rdm) > 1e-4, "For this ansatz, parameters, cost function should be one"
    
    graph_cost1 = GraphCyclWitness1Cost(ansatz=ansatz, instance = inst, N=N_graph, nb_params=N_params)
    cost1_opt = graph_cost1(X_SOL)
    cost1_rdm = graph_cost1(X_RDM)
    assert cost1_opt == 1.0, "For this ansatz, parameters, cost function should be one"
    assert  (fid_rdm - cost1_rdm) > 1e-4, "cost function1 should be lower than true fid"
    
    graph_cost2 = GraphCyclWitness2Cost(ansatz=ansatz, instance = inst, N=N_graph, nb_params=N_params)
    cost2_opt = graph_cost2(X_SOL)
    cost2_rdm = graph_cost2(X_RDM)
    assert cost2_opt == 1.0, "For this ansatz, parameters, cost function should be one"
    assert  (fid_rdm - cost2_rdm) > 1e-4, "cost function should be lower than true fid"
    
    
    