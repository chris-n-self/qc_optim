
__all__ = [
    'get_H2_qubit_op',
    'get_LiH_qubit_op',
    'run_BO_vqe',
    'run_BO_vqe_parallel',
]

import GPyOpt
import numpy as np

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries

from qiskit import Aer, execute, transpile, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator as wpo
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator
from qiskit.aqua.algorithms import ExactEigensolver

from .cost import *
from .ansatz import *
from . import utilities as ut

def get_H2_qubit_op(dist):
    """ """

    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), 
                         unit=UnitsType.ANGSTROM, 
                         charge=0, 
                         spin=0, 
                         basis='sto3g',
                        )
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type='parity', threshold=1E-8)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
    shift = repulsion_energy

    return qubitOp, num_particles, num_spin_orbitals, shift

def get_LiH_qubit_op(dist):
    """ """

    driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist), 
                         unit=UnitsType.ANGSTROM, 
                         charge=0, 
                         spin=0, 
                         basis='sto3g',
                        )
    molecule = driver.run()
    freeze_list = [0]
    remove_list = [-3, -2]
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type='parity', threshold=1E-8)
    #qubitOp = qubitOp.two_qubit_reduced_operator(num_particles)
    qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
    shift = repulsion_energy + energy_shift

    return qubitOp, num_particles, num_spin_orbitals, shift

def run_BO_vqe(dist,
               depth,
               molecule='H2',
               ansatz_type='xyz',
               seed=None,
               nb_iter=30,
               nb_init='max',
               init_jobs=1,
               nb_shots=1024,
               backend_name='qasm_simulator',
               initial_layout=None,
               optimization_level=3,
               verbose=False,
               **kwargs,
              ):
    """ 
    Run the BO VQE algorithm for a target molecule

    Parameters
    ----------
    distances : array
        The set of nuclear separations to run the BO VQE for
    depth : int
        The ansatz depth
    molecule : {'H2', 'LiH'}
        The molecule to run the BO VQE for
    ansatz_type : {'xyz', 'u3', 'random'}
        The pool executor used to compute the results.
    seed : int, optional 
        Passed to the random ansatz constructor for reproducibility

    nb_iter : int, default 30
        (BO) Sets the number of iteration rounds of the BO
    nb_init : int or keyword 'max', default 'max'
        (BO) Sets the number of initial data points to feed into the BO before starting
        iteration rounds. If set to 'max' it will generate the maximum number of initial
        points such that it submits `init_jobs` worth of circuits to a qiskit backend.
    init_jobs : int, default 1
        (BO) The number of qiskit jobs to use to generate initial data. (Most real device
        backends accept up to 900 circuits in one job.)

    nb_shots : int, default 1024
        (Qiskit) Number of measurements shots when executing circuits
    backend_name : name of qiskit backend, defaults to 'qasm_simulator'
        (Qiskit) Refers to either a IBMQ provider or Aer simulator backend
    initial_layout : int array, optional
        (Qiskit) Passed to a `QuantumInstance` obj, used in transpiling
    optimization_level : int, default 3
        (Qiskit) Passed to a `QuantumInstance` obj, used in transpiling

    verbose : bool, optional
        Set level of output of the function

    Returns
    -------
    BO_energies : array, size=len(distances)
        BO VQE estimatated energies of the molecular ground state at each distance
    exact_energies : array, size=len(distances)
        True energies of the molecular ground state at each distance
    """
    
    # parse molecule name argument
    if molecule=='H2':
        get_qubit_op = get_H2_qubit_op
    elif molecule=='LiH':
        get_qubit_op = get_LiH_qubit_op
    else:
        print('Molecule not recognised, please choose "H2" or "LiH".',file=sys.stderr)
        raise ValueError
    
    # make qubit ops
    qubitOp,num_particles,num_spin_orbitals,shift = get_qubit_op(dist)
    
    # get exact energy
    result = ExactEigensolver(qubitOp).run()
    exact_energy = result['energy']
    
    # make ansatz
    n = qubitOp.num_qubits
    if ansatz_type == 'random':
        ansatz = RandomAnsatz(n,depth,seed=seed)
    elif ansatz_type == 'xyz':
        ansatz = RegularXYZAnsatz(n,depth)
    elif ansatz_type == 'u3':
        ansatz = RegularU3Ansatz(n,depth) 
    else:
        print('ansatz_type not recognised.',file=sys.stderr)
        raise ValueError
                
    # create quantum instances
    if (backend_name[:4]=='ibmq'):
        _is_device = True
        
        from qiskit import IBMQ
        IBMQ.load_account()
        provider = IBMQ.get_provider(group='samsung')
        backend = provider.get_backend(backend_name)
    else:
        _is_device = False
        backend = Aer.get_backend(backend_name)
    inst_fewshots = QuantumInstance(backend, 
                                    shots=nb_shots, 
                                    optimization_level=optimization_level, 
                                    initial_layout=initial_layout,
                                    skip_qobj_validation=(not _is_device),
                                   )
    inst_bigshots = QuantumInstance(backend, 
                                    shots=8192, 
                                    optimization_level=optimization_level, 
                                    initial_layout=initial_layout,
                                    skip_qobj_validation=(not _is_device),
                                   )
    
    # make Cost objs
    cost_fewshots = CostWeightedOps(ansatz=ansatz, instance=inst_fewshots, operators=qubitOp, verbose=False)
    cost_bigshots = CostWeightedOps(ansatz=ansatz, instance=inst_bigshots, operators=qubitOp, verbose=False)
    
    # ===================
    # run BO Optim
    # ===================
    # setup
    DOMAIN_FULL = [(0, 2*np.pi) for i in range(ansatz.nb_params)]
    DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
    bo_args = ut.gen_default_argsbo()
    bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':0})
    cost_bo = cost_fewshots

    # get initial values using bigshots
    if nb_init=='max':
        _nb_init = init_jobs*900//cost_fewshots.num_circuits
    else:
        _nb_init = nb_init
    if verbose: print('initialising...')
    Xinit = 2*np.pi*np.random.random(_nb_init*ansatz.nb_params).reshape((_nb_init,ansatz.nb_params))
    Yinit = cost_fewshots(Xinit)
    
    # optim
    if verbose: print('optimising...')
    Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, 
                                               X=Xinit, 
                                               Y=Yinit, 
                                               **bo_args)
    Bopt.run_optimization(max_iter=nb_iter, eps=0)

    # Results found
    (Xseen, Yseen), (Xexp,Yexp) = Bopt.get_best()
    if verbose:
        print("best seen: "+f'{cost_bigshots(Xseen)}')
        print("best expected: "+f'{cost_bigshots(Xexp)}')
        print(Bopt.model.model)
        Bopt.plot_convergence()
    print(f'{np.real(cost_bigshots(Xexp)+shift)[0]}'+' '+f'{exact_energy+shift}')
    
    return np.real(cost_bigshots(Xexp)+shift)[0],exact_energy+shift

def run_BO_vqe_parallel(distances,
                        depth,
                        molecule='H2',
                        ansatz_type='xyz',
                        seed=None,
                        nb_iter=30,
                        nb_init='max',
                        init_jobs=1,
                        nb_shots=1024,
                        backend_name='qasm_simulator',
                        initial_layout=None,
                        optimization_level=3,
                        verbose=False,
                        **kwargs,
                        ):
    """ 
    Run the BO VQE algorithm, parallelised over different nuclear separations

    Parameters
    ----------
    distances : array
        The set of nuclear separations to run the BO VQE for
    depth : int
        The ansatz depth
    molecule : {'H2', 'LiH'}
        The molecule to run the BO VQE for
    ansatz_type : {'xyz', 'u3', 'random'}
        The pool executor used to compute the results.
    seed : int, optional 
        Passed to the random ansatz constructor for reproducibility

    nb_iter : int, default 30
        (BO) Sets the number of iteration rounds of the BO
    nb_init : int or keyword 'max', default 'max'
        (BO) Sets the number of initial data points to feed into the BO before starting
        iteration rounds. If set to 'max' it will generate the maximum number of initial
        points such that it submits `init_jobs` worth of circuits to a qiskit backend.
    init_jobs : int, default 1
        (BO) The number of qiskit jobs to use to generate initial data. (Most real device
        backends accept up to 900 circuits in one job.)

    nb_shots : int, default 1024
        (Qiskit) Number of measurements shots when executing circuits
    backend_name : name of qiskit backend, defaults to 'qasm_simulator'
        (Qiskit) Refers to either a IBMQ provider or Aer simulator backend
    initial_layout : int array, optional
        (Qiskit) Passed to a `QuantumInstance` obj, used in transpiling
    optimization_level : int, default 3
        (Qiskit) Passed to a `QuantumInstance` obj, used in transpiling

    verbose : bool, optional
        Set level of output of the function

    Returns
    -------
    exact_energies : array, size=len(distances)
        True energies of the molecular ground state at each distance
    BO_energies : array, size=len(distances)
        BO VQE estimatated energies of the molecular ground state at each distance
    BO_energies_std : array, size=len(distances)
        The std on the final estimate of BO_energies. This comes from the std of the 
        final circuit evaluation at the BO optimal, it does not include uncertainties
        due to the BO's lack of confidence about its optimal
    optimal_params : array, shape=(len(distances),nb_params)
        Optimal parameter sets found for each separation
    """
    
    # parse molecule name argument
    if molecule=='H2':
        get_qubit_op = get_H2_qubit_op
    elif molecule=='LiH':
        get_qubit_op = get_LiH_qubit_op
    else:
        print('Molecule not recognised, please choose "H2" or "LiH".',file=sys.stderr)
        raise ValueError

    # make qubit ops
    if verbose: print('making qubit ops...')
    qubit_ops,exact_energies,shifts = _make_qubit_ops(distances,get_qubit_op)
    # group pauli operators for measurement
    qubit_ops = [ TPBGroupedWeightedPauliOperator.unsorted_grouping(op) for op in qubit_ops ]
                
    # make ansatz
    n = qubit_ops[0].num_qubits
    if ansatz_type == 'random':
        ansatz = RandomAnsatz(n,depth,seed=seed)
    elif ansatz_type == 'xyz':
        ansatz = RegularXYZAnsatz(n,depth)
    elif ansatz_type == 'u3':
        ansatz = RegularU3Ansatz(n,depth) 
    else:
        print('ansatz_type not recognised.',file=sys.stderr)
        raise ValueError

    # create quantum instances
    if (backend_name[:4]=='ibmq'):
        _is_device = True
        
        from qiskit import IBMQ
        IBMQ.load_account()
        provider = IBMQ.get_provider(group='samsung')
        backend = provider.get_backend(backend_name)
    else:
        _is_device = False
        backend = Aer.get_backend(backend_name)
    inst_fewshots = QuantumInstance(backend, 
                                    shots=nb_shots, 
                                    optimization_level=optimization_level, 
                                    initial_layout=initial_layout,
                                    skip_qobj_validation=(not _is_device),
                                   )
    inst_bigshots = QuantumInstance(backend, 
                                    shots=8192, 
                                    optimization_level=optimization_level, 
                                    initial_layout=initial_layout,
                                    skip_qobj_validation=(not _is_device),
                                   )
    
    # generate and transpile measurement circuits, using first of qubit_ops
    measurement_circuits = qubit_ops[0].construct_evaluation_circuit(
        wave_function=ansatz.circuit,
        statevector_mode=inst_fewshots.is_statevector)
    t_measurement_circuits = inst_fewshots.transpile(measurement_circuits)

    def obtain_results(params_values,quantum_instance):
        """
        Function to wrap executions on the quantum backend, binds parameter values
        and executes.

        Gets `ansatz` and `_t_measurement_circuits` from the surrounding scope. 
        `quantum_instance` is an arg because we switch between high and low shot 
        number instances.
        """

        if np.ndim(params_values)==1:
            params_values = [params_values]

        # package and bind circuits
        bound_circs = []
        for pidx,p in enumerate(params_values):
            for cc in t_measurement_circuits:
                tmp = cc.bind_parameters(dict(zip(ansatz.params, p)))
                tmp.name = str(pidx) + tmp.name
                bound_circs.append(tmp)
            
        # See if one can add a noise model here and the number of parameters
        return quantum_instance.execute(bound_circs, had_transpiled=True)

    # ===================
    # run BO Optim
    # ===================

    # setup
    DOMAIN_FULL = [(0, 2*np.pi) for i in range(ansatz.nb_params)]
    DOMAIN_BO = [{'name': str(i), 'type': 'continuous', 'domain': d} for i, d in enumerate(DOMAIN_FULL)]
    bo_args = ut.gen_default_argsbo()
    bo_args.update({'domain': DOMAIN_BO,'initial_design_numdata':0})

    # get initial values using bigshots
    if nb_init=='max':
        _nb_init = init_jobs*900//len(t_measurement_circuits)
    else:
        _nb_init = nb_init
    if verbose: print('getting initialisation data...')
    Xinit = 2*np.pi*np.random.random(_nb_init*ansatz.nb_params).reshape((_nb_init,ansatz.nb_params))
    init_results = obtain_results(Xinit,inst_bigshots)
    
    # initialise BO obj for each distance
    Bopts = []
    _update_weights = []
    if verbose: print('initialising BO objects...')
    for qo in qubit_ops:
        # evaluate this distance's specific weighted qubit operators using the results
        Yinit = np.array([[ np.real(qo.evaluate_with_result(init_results,
            statevector_mode=inst_bigshots.is_statevector,
            circuit_name_prefix=str(i))[0]) for i in range(Xinit.shape[0]) ]]).T 
        tmp = GPyOpt.methods.BayesianOptimization(lambda x: None, # blank cost function
                                                  X=Xinit, 
                                                  Y=Yinit, 
                                                  **bo_args)
        tmp.run_optimization(max_iter=0, eps=0) # (may not be needed)
        Bopts.append(tmp)

        # This block is only to ensure the linear decrease of the exploration
        if(getattr(tmp, '_dynamic_weights') == 'linear'):
            _update_weights.append(True)
            def dynamics_weight(n):
                return max(0.000001, bo_args['acquisition_weight']*(1 - n/nb_iter))
        else:
            _update_weights.append(False)

    # main loop
    for iter_idx in range(nb_iter):

        # for the last 5 shots use large number of measurements
        inst = inst_fewshots
        _msg = 'at optimisation round '+f'{iter_idx}'
        if nb_iter-iter_idx<6:
            inst = inst_bigshots
            _msg += ' (using big shot number)'
        if verbose: print(_msg)

        # query each BO obj where it would like an evaluation, and pool them all togther
        for idx,bo in enumerate(Bopts):
            bo._update_model(bo.normalization_type)
            if(_update_weights[idx]):
                bo.acquisition.exploration_weight = dynamics_weight(iter_idx)
            x = bo._compute_next_evaluations()
            if idx==0:
                Xnew = x
            else:
                Xnew = np.vstack((Xnew,x))
        
        # get results object at new param values
        new_results = obtain_results(Xnew,inst)

        # iterate over the BO obj's passing them all the correctly weighted data
        for bo,qo in zip(Bopts,qubit_ops):
            Ynew = np.array([[ np.real(qo.evaluate_with_result(new_results,
                statevector_mode=inst_bigshots.is_statevector,
                circuit_name_prefix=str(i))[0]) for i in range(Xnew.shape[0]) ]]).T 
            bo.X = np.vstack((bo.X, Xnew))
            bo.Y = np.vstack((bo.Y, Ynew))

    #finalize (may not be needed)
    if verbose: print('finalising...')
    for bo in Bopts:
        bo.run_optimization(max_iter = 0, eps = 0)
    
    # Results found
    for idx,(dist,bo) in enumerate(zip(distances,Bopts)):
        (Xseen, Yseen), (Xexp,Yexp) = bo.get_best()
        if verbose:
            print("at distance "+f'{dist}')
            print("best seen: "+f'{Yseen}')
            print("best expected: "+f'{Yexp}')
            print(bo.model.model)
            bo.plot_convergence()    
        if idx==0:
            Xfinal = Xexp
        else:
            Xfinal = np.vstack([Xfinal,Xexp])
    
    # obtain final BO estimates from an evaluation of the optimal params
    final_results = obtain_results(Xfinal,inst_bigshots)
    BO_energies = np.zeros(len(distances))
    BO_energies_std = np.zeros(len(distances))
    for idx,qo in enumerate(qubit_ops):
        mean,std = qo.evaluate_with_result(final_results,
            statevector_mode=inst_bigshots.is_statevector,
            circuit_name_prefix=str(idx))
        BO_energies[idx] = np.real(mean)
        BO_energies_std[idx] = np.real(std)
    
    return exact_energies+shifts,BO_energies+shifts,BO_energies_std,Xfinal

def _make_qubit_ops(distances,qubit_op_func):
    """
    Make qubit ops for `run_BO_vqe_parallel`, need to ensure all the distances have the
    same set of (differently weighted) Pauli operators. Also get the exact ground state 
    energies here.
    
    Parameters
    ----------
    distances : array
        The set of nuclear separations

    Returns
    -------
    qubit_ops : array of WeightedPauliOperator objs, size=len(distances)
        The set of WeightedPauliOperator corresponding to each distance
    exact_energies : array, size=len(distances)
        UNSHIFTED true energies of the molecular ground state at each distance
    shifts : array, size=len(distances)
        Molecular ground state shifts at each distance
    """
    qubit_ops = []
    exact_energies = np.zeros(len(distances))
    shifts = np.zeros(len(distances))
    for idx,dist in enumerate(distances):
        qubitOp,num_particles,num_spin_orbitals,shift = qubit_op_func(dist)
        shifts[idx] = shift

        if idx>0:
            if not len(qubitOp.paulis)==len(test_pauli_set):
                # the new qubit op has a different number of Paulis than the previous
                new_pauli_set = set([ p[1] for p in qubitOp.paulis ])
                if len(qubitOp.paulis)>len(test_pauli_set):
                    # the new operator set has more paulis the previous
                    missing_paulis = list(new_pauli_set - test_pauli_set)
                    paulis_to_add = [ [qubitOp.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # iterate over previous qubit ops and add new paulis
                    for prev_op in qubit_ops:
                        prev_op.add(wpo_to_add)
                    # save new reference pauli set
                    test_pauli_set = new_pauli_set
                else:
                    # the new operator set has less paulis than the previous
                    missing_paulis = list(test_pauli_set - new_pauli_set)
                    paulis_to_add = [ [qubitOp.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # add new paulis to current qubit op
                    qubitOp.add(wpo_to_add)
        else:
            test_pauli_set = set([ p[1] for p in qubitOp.paulis ])

        # get exact energy
        result = ExactEigensolver(qubitOp).run()
        exact_energies[idx] = result['energy']

        qubit_ops.append(qubitOp)

    return qubit_ops,exact_energies,shifts
