
# list of * contents
__all__ = ['CostWeightedOps']

import qiskit as qk
import numpy as np
import pdb
#import itertools as it
pi = np.pi

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


if __name__ == '__main__':    

    import GPyOpt
    import numpy as np
    import utilities as ut
    from ansatz import *

    # Create an instance
    sim = qk.Aer.get_backend('qasm_simulator')
    inst_fewshots = qk.aqua.QuantumInstance(sim, shots=1024, optimization_level=3)
    inst_bigshots = qk.aqua.QuantumInstance(sim, shots=8192, optimization_level=3)

    # make H2 qubit Hamiltonian
    def get_qubit_op(dist):
        """ """
        from qiskit.chemistry import FermionicOperator
        from qiskit.chemistry.drivers import PySCFDriver, UnitsType
        from qiskit.aqua.operators import Z2Symmetries
        driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 "+str(dist), 
                             unit=UnitsType.ANGSTROM, 
                             charge=0, 
                             spin=0, 
                             basis='sto3g')
        molecule = driver.run()
        repulsion_energy = molecule.nuclear_repulsion_energy
        num_particles = molecule.num_alpha + molecule.num_beta
        num_spin_orbitals = molecule.num_orbitals * 2
        ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
        qubitOp = ferOp.mapping(map_type='parity', threshold=1E-8)
        qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
        shift = repulsion_energy
        return qubitOp, num_particles, num_spin_orbitals, shift
    dist = 0.7
    qubitOp,num_particles,num_spin_orbitals,shift = get_qubit_op(dist)

    # make ansatz
    n = num_particles
    depth = 3
    ansatz = RandomAnsatz(n,depth)

    # make Cost objs
    cost_fewshots = CostWeightedOps(ansatz=ansatz, instance=inst_fewshots, operators=qubitOp)
    cost_bigshots = CostWeightedOps(ansatz=ansatz, instance=inst_bigshots, operators=qubitOp)

    # ===================
    # BO Optim
    # No noise / Use of fidelity
    # ===================
    # setup
    NB_INIT = 30
    NB_ITER = 30
    DOMAIN_FULL = [(0, 2*np.pi) for i in range(ansatz.nb_params)]
    DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
    bo_args = ut.gen_default_argsbo()
    bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':NB_INIT})
    cost_bo = cost_fewshots

    #optim
    Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, **bo_args)    
    print("start optim")
    Bopt.run_optimization(max_iter = NB_ITER, eps = 0)

    # Results found
    (x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
    cost_bigshots(x_seen)
    cost_bigshots(x_exp)
    print(Bopt.model.model)
    Bopt.plot_convergence()
