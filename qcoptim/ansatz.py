
# list of * contents
__all__ = ['ParameterisedAnsatz',
           'RandomAnsatz']

import random
import qiskit as qk

class ParameterisedAnsatz(object):
    """ """

    def __init__(
                 self,
                 num_qubits,
                 depth,
                 **kwargs
                ):
        self.num_qubits = num_qubits
        self.depth = depth

        # make circuit and generate parameters
        self.params = self._generate_params()
        self.nb_params = len(self.params)
        self.circuit = self._generate_circuit()

    def _generate_params(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

    def _generate_circuit(self):
        """ To be implemented in the subclasses """
        raise NotImplementedError

class RandomAnsatz(ParameterisedAnsatz):
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
        nb_params = 2*(self.num_qubits-1)*self.depth + self.num_qubits
        name_params = ['R'+str(i) for i in range(nb_params)]
        return [qk.circuit.Parameter(n) for n in name_params]

    def _generate_circuit(self):
        """ """

        # the set number of entangling pairs to distribute randomly
        ent_pairs = [(i, i + 1) for i in range(self.num_qubits - 1) for _ in range(self.depth)]
        random.shuffle(ent_pairs)
        
        # keep track of where not to apply a entangling gate again
        just_entangled = set()
            
        # keep track of where its worth putting a parameterised gate
        needs_rgate = [True] * self.num_qubits

        # make circuit obj and list of parameter obj's created
        qc = qk.QuantumCircuit(self.num_qubits)

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
            for i in range(self.num_qubits):
                if needs_rgate[i]:
                    (single_qubit_gates[random.randint(0,2)])(self.params[param_counter],i)
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
        
        for i in range(self.num_qubits):
            if needs_rgate[i]:
                (single_qubit_gates[random.randint(0,2)])(self.params[param_counter],i)
                param_counter += 1
        
        return qc
