
# list of * contents
__all__ = [
    'BaseAnsatz',
    'RandomAnsatz',
    'RegularXYZAnsatz',
    'RegularU3Ansatz',
]

import random
import qiskit as qk
import numpy as np

from abc import ABC, abstractmethod

class ParameterisedAnsatz(ABC):
    """
    Interface for a parameterised ansatz. Specifies an object that must have four
    properties: a depth, a circuit, the number of qubits and the number of parameters.
    """

    @property
    @abstractmethod
    def depth(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def circuit(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def nb_params(self):
        raise NotImplementedError

class BaseAnsatz(ParameterisedAnsatz):
    """ """

    def __init__(
                 self,
                 num_qubits,
                 depth,
                 **kwargs
                ):
        self._num_qubits = num_qubits
        self._depth = depth

        # make circuit and generate parameters
        self._params = self._generate_params()
        self._nb_params = len(self._params)
        self._circuit = self._generate_circuit()

    def _generate_params(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

    def _generate_circuit(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

    @property
    def depth(self):
        return self._depth

    @property
    def circuit(self):
        return self._circuit

    @property
    def params(self):
        return self._params

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def nb_params(self):
        return self._nb_params

class RandomAnsatz(BaseAnsatz):
    """ """

    def __init__(self,*args,
                 seed=None,
                 gate2='CX',
                ):

        # set random seed if passed
        if seed is not None:
            random.seed(seed)

        # set two-qubit gate
        self.gate2 = gate2

        # explicitly call base class initialiser
        super(RandomAnsatz,self).__init__(*args)

    def _generate_params(self):
        """ """
        nb_params = 2*(self._num_qubits-1)*self._depth + self._num_qubits
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        # the set number of entangling pairs to distribute randomly
        ent_pairs = [(i, i + 1) for i in range(self._num_qubits - 1) for _ in range(self._depth)]
        random.shuffle(ent_pairs)
        
        # keep track of where not to apply a entangling gate again
        just_entangled = set()
            
        # keep track of where its worth putting a parameterised gate
        needs_rgate = [True] * self._num_qubits

        # make circuit obj and list of parameter obj's created
        qc = qk.QuantumCircuit(self._num_qubits)

        # parse entangling gate arg
        if self.gate2=='CZ':
            ent_gate = qc.cz
        elif self.gate2=='CX':
            ent_gate = qc.cx
        else:
            print("entangling gate not recognised, please specify: 'CX' or 'CZ'", file=sys.stderr)
            raise ValueError
            
        # array of single qubit e^{-i\theta/2 \sigma_i} type gates, we will
        # randomly draw from these
        single_qubit_gates = [qc.rx,qc.ry,qc.rz]
        
        # track next parameter to use
        param_counter = 0

        # consume list of pairs to entangle
        while ent_pairs:
            for i in range(self._num_qubits):
                if needs_rgate[i]:
                    (single_qubit_gates[random.randint(0,2)])(self._params[param_counter],i)
                    param_counter += 1
                    needs_rgate[i] = False
                    
            for k, pair in enumerate(ent_pairs):
                if pair not in just_entangled:
                    break
            i, j = ent_pairs.pop(k)
            ent_gate(i, j)
            
            just_entangled.add((i, j))
            just_entangled.discard((i - 1, j - 1))
            just_entangled.discard((i + 1, j + 1))
            needs_rgate[i] = needs_rgate[j] = True
        
        for i in range(self._num_qubits):
            if needs_rgate[i]:
                (single_qubit_gates[random.randint(0,2)])(self._params[param_counter],i)
                param_counter += 1
        
        return qc

class RegularXYZAnsatz(BaseAnsatz):
    """ """

    def _generate_params(self):
        """ """
        nb_params = self._num_qubits*(self._depth+1)
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._num_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)
        
        egate = qc.cx # entangle with CNOTs
        single_qubit_gate_sequence = [qc.rx,qc.ry,qc.rz] # eisert scheme alternates RX, RY, RZ 
        
        # initial round in the Eisert scheme is fixed RY rotations at 
        # angle pi/4
        qc.ry(np.pi/4,range(N))
        l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
        if len(l)==1:
            egate(l[0],r[0])
        elif len(l)>1:
            egate(l,r)
        l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
        if len(l)==1:
            egate(l[0],r[0])
        elif len(l)>1:
            egate(l,r)
        if barriers:
            qc.barrier()
        
        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):
                gate = single_qubit_gate_sequence[r % len(single_qubit_gate_sequence)]
                gate(self._params[param_counter],q)
                param_counter += 1

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()
        
        # add final round of parameterised single qubit rotations
        for q in range(N):
            gate = single_qubit_gate_sequence[self._depth % len(single_qubit_gate_sequence)]
            gate(self._params[param_counter],q)
            param_counter += 1

        return qc

class RegularU3Ansatz(BaseAnsatz):
    """ """

    def _generate_params(self):
        """ """
        nb_params = self._num_qubits*(self._depth+1)*3
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        N = self._num_qubits
        barriers = True
        
        qc = qk.QuantumCircuit(N)

        egate = qc.cx # entangle with CNOTs

        param_counter = 0
        for r in range(self._depth):

            # add parameterised single qubit rotations
            for q in range(N):
                qc.u3(*[self._params[param_counter+i] for i in range(3)],q)
                param_counter += 3

            # add entangling gates
            l,r = 2*np.arange(N//2),2*np.arange(N//2)+1
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            l,r = 2*np.arange(N//2-1+(N%2))+1,2*np.arange(N//2-1+(N%2))+2
            if len(l)==1:
                egate(l[0],r[0])
            elif len(l)>1:
                egate(l,r)
            if barriers:
                qc.barrier()

        # add final round of parameterised single qubit rotations
        for q in range(N):
            qc.u3(*[self._params[param_counter+i] for i in range(3)],q)
            param_counter += 3
        
        return qc