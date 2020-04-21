
__all__ = [
    'get_H2_qubit_op',
    'get_LiH_qubit_op',
    'get_TFIM_qubit_op',
    'run_BO_vqe',
    'run_BO_vqe_parallel',
]

import time
import json
import GPyOpt
import numpy as np

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries

from qiskit import Aer, execute, transpile, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator as wpo
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator
from qiskit.aqua.algorithms import ExactEigensolver

from .cost import *
from .ansatz import *
from . import utilities as ut

def get_H2_qubit_op(dist):
    """ 
    Use the qiskit chemistry package to get the qubit Hamiltonian for LiH

    Parameters
    ----------
    dist : float
        The nuclear separations

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian
    num_particles : int
        Number of electrons that are not frozen
    num_spin_orbitals : int
        Number of spin orbitals (2x atomic orbitals) that are not frozen
    shift : float
        The ground state of the qubit Hamiltonian needs to be corrected by this amount of
        energy to give the real physical energy. This includes the replusive energy between
        the nuclei and the energy shift of the frozen orbitals.
    """
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
    """ 
    Use the qiskit chemistry package to get the qubit Hamiltonian for LiH

    Parameters
    ----------
    dist : float
        The nuclear separations

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian
    num_particles : int
        Number of electrons that are not frozen
    num_spin_orbitals : int
        Number of spin orbitals (2x atomic orbitals) that are not frozen
    shift : float
        The ground state of the qubit Hamiltonian needs to be corrected by this amount of
        energy to give the real physical energy. This includes the replusive energy between
        the nuclei and the energy shift of the frozen orbitals.
    """
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

def get_TFIM_qubit_op(
    N,
    B,
    J=1,
    pbc=False,
    ):
    """ 
    Construct the qubit Hamiltonian for 1d TFIM: H = \sum_{i} ( J Z_i Z_{i+1} + B X_i ), 
    matching the return pattern of the chemistry cases above

    Parameters
    ----------
    N : int
        The number of spin 1/2 particles in the chain
    B : float
        Transverse field strength
    J : float, optional default 1.
        Ising interaction strength
    pbc : boolean, optional default False
        Set the boundary conditions of the 1d spin chain

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian
    BLANK : None
        Blank to match returns of chemistry functions
    BLANK : None
        Blank to match returns of chemistry functions
    BLANK : 0.
        Blank to match returns of chemistry functions
    """

    pauli_terms = []
    # ZZ terms
    pauli_terms += [ (J,Pauli.from_label('I'*(i)+'ZZ'+'I'*((N-1)-(i+1)))) for i in range(N-1) ]
    # optional periodic boundary condition term
    if pbc:
        pauli_terms += [ (J,Pauli.from_label('Z'+'I'*(N-2)+'Z')) ]
    # X terms
    pauli_terms += [ (B,Pauli.from_label('I'*(i)+'X'+'I'*(N-(i+1)))) for i in range(N) ]

    qubitOp = wpo(pauli_terms)

    return qubitOp,None,None,0.


def run_BO_vqe(
    dist,
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
    dist : float
        The nuclear separations to run the BO VQE for
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

def run_BO_vqe_parallel(
    phys_params,
    depth,
    molecule='H2',
    N=None,
    J=None,
    pbc=None,
    ansatz_type='xyz',
    seed=None,
    info_sharing_mode='shared',
    nb_iter=30,
    nb_init='max',
    init_jobs=1,
    nb_shots=1024,
    backend_name='qasm_simulator',
    initial_layout=None,
    optimization_level=3,
    verbose=False,
    dump_results=False,
    results_directory='.',
    xyzpy_max_nb_params=None,
    **kwargs,
    ):
    """ 
    Run the BO VQE algorithm, parallelised over different nuclear separations

    Parameters
    ----------
    phys_params : array
        The set of physical parameters to run the BO VQE for. In the chemistry case this
        is the nuclear separations, for TFIM it is the set of B values
    depth : int
        The ansatz depth

    molecule : {'H2', 'LiH', 'TFIM'}
        The physical system to run the BO VQE for
    N : None or int
        This will be ignored for molecules, needed for TFIM generator
    J : None or float
        This will be ignored for molecules, optionally passed to TFIM generator
    pbc : None or boolean
        This will be ignored for molecules, optionally passed to TFIM generator

    ansatz_type : {'xyz', 'u3', 'random'}
        Ansatz type, refers to the classes in the ansatz module
    seed : int, optional 
        Passed to the random ansatz constructor for reproducibility
    
    info_sharing_mode : {'shared','random','left','right'}
        (BO) This controls the evaluation sharing of the BO instances, cases:
            'shared' :  Each BO obj gains access to evaluations of all of the others. 
            'random1' : The BO do not get the evaluations others have requested, but in 
                addition to their own they get an equivalent number of randomly chosen 
                parameter points 
            'random2' : The BO do not get the evaluations others have requested, but in 
                addition to their own they get an equivalent number of randomly chosen 
                parameter points. These points are not chosen fully at random, but instead 
                if x1 and x2 are BO[1] and BO[2]'s chosen evaluations respectively then BO[1] 
                get an additional point y2 that is |x2-x1| away from x1 but in a random 
                direction, similar for BO[2], etc.
            'left', 'right' : Implement information sharing but in a directional way, so 
                that (using 'left' as an example) BO[1] gets its evaluation as well as 
                BO[0]; BO[2] gets its point as well as BO[1] and BO[0], etc. To ensure all 
                BO's get an equal number of evaluations this is padded with random points. 
                These points are not chosen fully at random, they are chosen in the same way
                as 'random2' described above.
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
    dump_results : bool, default False
        Flag for whether or not to dump the accumulated results objs
    results_directory : string (optional)
        Set a relative path to the directory to be used for dumping results objs

    xyzpy_max_nb_params : None or int, (hack)
        (xyzpy) Number of params varies with ansatz depth, this conflicts with the xarrays
        used to store output if the Returns includes the optimal parameter set. This pads
        the number of parameters out to some max
    **kwargs : additional kwargs
        (xyzpy) This is mostly here to allow additional fields to be added in xyzpy

    Returns
    -------
    exact_energies : array, size=len(phys_params)
        True energies of the molecular ground state at each distance
    BO_energies : array, size=len(phys_params)
        BO VQE estimatated energies of the molecular ground state at each distance
    BO_energies_std : array, size=len(phys_params)
        The std on the final estimate of BO_energies. This comes from the std of the 
        final circuit evaluation at the BO optimal, it does not include uncertainties
        due to the BO's lack of confidence about its optimal
    optimal_params : array, shape=(len(phys_params),nb_params)
        Optimal parameter sets found for each separation
    results_dump_filename : None, or string
        If `dump_results` is set to True this will return the relative filepath of the
        location the results have been dumped to
    """

    # to save results sets
    accumulated_results = []
    
    # parse molecule name argument
    if molecule=='H2':
        get_qubit_op = get_H2_qubit_op
    elif molecule=='LiH':
        get_qubit_op = get_LiH_qubit_op
    elif molecule=='TFIM':
        # check we have N, J, pbc args (N is essential, J and pbc are optional args of
        # get_TFIM_qubit_op, if they are None here we do not pass them to preserve the
        # defaults of get_TFIM_qubit_op)
        if not isinstance(N,(int,np.integer)):
            print('TFIM generator was passed invalid N arg: '+f'{N}',file=sys.stderr)
            raise ValueError
        _tfim_wrapper_args = {}
        if not J is None:
            _tfim_wrapper_args['J'] = J
        if not pbc is None:
            _tfim_wrapper_args['pbc'] = pbc
        # make get_qubit_op func by wrapping get_TFIM_qubit_op function
        def get_qubit_op(B):
            return get_TFIM_qubit_op(N,B,**_tfim_wrapper_args)
    else:
        print('Molecule not recognised, please choose "H2", "LiH" or "TFIM".',file=sys.stderr)
        raise ValueError

    # check the information sharing arg is recognised
    if not info_sharing_mode in ['shared','random1','random2','left','right']:
        print('BO information sharing mode '+f'{info_sharing_mode}'+' not recognised, please choose: '
            +'"shared", "random1", "random2", "left" or "right".',file=sys.stderr)
        raise ValueError

    # make qubit ops
    if verbose: print('making qubit ops...')
    qubit_ops,exact_energies,shifts = _make_qubit_ops(phys_params,get_qubit_op)
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
        print('ansatz_type '+f'{ansatz_type}'+' not recognised, please choose:'
            +'"xyz", "u3" or "random"',file=sys.stderr)
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

    def obtain_results(params_values,quantum_instance,circuit_name_prefixes=None):
        """
        Function to wrap executions on the quantum backend, binds parameter values
        and executes.
        (Optionally `circuit_name_prefixes` arg sets the prefix names of the eval
        circuits, else they are numbered sequentially. If the size of the name array
        is different from the size of `params_values` it will crash.)

        Gets `ansatz` and `t_measurement_circuits` from the surrounding scope. 
        `quantum_instance` is an arg because we switch between high and low shot 
        number instances.
        """
        if np.ndim(params_values)==1:
            params_values = [params_values]

        if circuit_name_prefixes is not None:
            assert len(circuit_name_prefixes)==params_values.shape[0]

        # package and bind circuits
        bound_circs = []
        for pidx,p in enumerate(params_values):
            for cc in t_measurement_circuits:
                tmp = cc.bind_parameters(dict(zip(ansatz.params, p)))
                if circuit_name_prefixes is not None:
                    prefix = circuit_name_prefixes[pidx]
                else:
                    prefix = str(pidx)
                tmp.name = prefix + tmp.name
                bound_circs.append(tmp)

        # See if one can add a noise model here and the number of parameters
        result = quantum_instance.execute(bound_circs, had_transpiled=True)

        if dump_results:
            accumulated_results.append([params_values.tolist(),result.to_dict()])
            
        return result

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
        circuit_name_prefixes = []
        circ_name_to_x_map = {}
        for idx,bo in enumerate(Bopts):
            bo._update_model(bo.normalization_type)
            if(_update_weights[idx]):
                bo.acquisition.exploration_weight = dynamics_weight(iter_idx)
            x = bo._compute_next_evaluations()
            if idx==0:
                Xnew = x
            else:
                Xnew = np.vstack((Xnew,x))
                
            _circ_name = str(idx)+'-base'
            circuit_name_prefixes.append(_circ_name)
            circ_name_to_x_map[_circ_name] = x[0]

            # to implement 'random1' we pad out each with completely random evaluations
            if info_sharing_mode=='random1':
                nb_extra_points = len(phys_params) - 1
                extra_points = 2*np.pi*np.random.random(nb_extra_points*ansatz.nb_params).reshape((nb_extra_points,ansatz.nb_params))
                Xnew = np.vstack((Xnew,extra_points))
                
                _circ_names = [ str(idx)+'-'+str(i) for i in range(len(phys_params)) if not i==idx ]
                circuit_name_prefixes += _circ_names
                circ_name_to_x_map.update(dict(zip(_circ_names,extra_points)))

        # to implement 'random2','left','right' strategies we have to wait until all primary
        # evaluations have been processed
        tmp = copy.deepcopy(Xnew)
        if info_sharing_mode in ['random2','left','right']:
            for boidx,bo in enumerate(Bopts):
                bo_eval_point = tmp[boidx]

                for pidx,p in enumerate(tmp): 
                    if (
                        ((info_sharing_mode=='random2') and (not boidx==pidx))
                        or ((info_sharing_mode=='left') and (boidx<pidx))
                        or ((info_sharing_mode=='right') and (boidx>pidx))
                       ):
                        dist = np.sqrt(np.sum((bo_eval_point-p)**2)) # L2 norm distance
                        # generate random vector in N-d space then scale it to have length we want, 
                        # using 'Hypersphere Point Picking' Gaussian approach
                        random_displacement = np.random.normal(size=ansatz.nb_params)
                        random_displacement = random_displacement * dist/np.sqrt(np.sum(random_displacement**2))
                        Xnew = np.vstack((Xnew,bo_eval_point+random_displacement))
                        
                        _circ_name = str(boidx)+'-'+str(pidx)
                        circuit_name_prefixes.append(_circ_name)
                        circ_name_to_x_map[_circ_name] = bo_eval_point+random_displacement

        # sense check on number of circuits generated
        if info_sharing_mode=='shared':
            assert len(circuit_name_prefixes)==len(phys_params)
        elif info_sharing_mode in ['random1','random2']:
            assert len(circuit_name_prefixes)==len(phys_params)**2
        elif info_sharing_mode in ['left','right']:
            assert len(circuit_name_prefixes)==len(phys_params)*(len(phys_params)+1)//2
        
        # get results object at new param values
        new_results = obtain_results(Xnew,inst,circuit_name_prefixes=circuit_name_prefixes)

        # iterate over the BO obj's passing them all the correctly weighted data
        for idx,(bo,qo) in enumerate(zip(Bopts,qubit_ops)):

            if info_sharing_mode=='shared':
                _pull_from = [ str(i)+'-base' for i in range(len(phys_params)) ]
            elif info_sharing_mode in ['random1','random2']:
                _pull_from = ([ str(idx)+'-base' ]
                    + [ str(idx)+'-'+str(i) for i in range(len(phys_params)) if not i==idx ])
            elif info_sharing_mode=='left':
                _pull_from = ([ str(i)+'-base' for i in range(idx+1) ] 
                    + [ str(idx)+'-'+str(i) for i in range(idx+1,len(phys_params)) ])
            elif info_sharing_mode=='right':
                _pull_from = ([ str(i)+'-base' for i in range(idx,len(phys_params)) ] 
                    + [ str(idx)+'-'+str(i) for i in range(idx) ])

            assert len(_pull_from)==len(phys_params) # sense check

            Ynew = np.array([[ np.real(qo.evaluate_with_result(new_results,
                statevector_mode=inst_bigshots.is_statevector,
                circuit_name_prefix=i)[0]) for i in _pull_from ]]).T 
            Xnew = np.array([ circ_name_to_x_map[i] for i in _pull_from ])
            
            bo.X = np.vstack((bo.X, Xnew))
            bo.Y = np.vstack((bo.Y, Ynew))

    #finalize (may not be needed)
    if verbose: print('finalising...')
    for bo in Bopts:
        bo.run_optimization(max_iter = 0, eps = 0)
    
    # Results found
    for idx,(dist,bo) in enumerate(zip(phys_params,Bopts)):
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
    BO_energies = np.zeros(len(phys_params))
    BO_energies_std = np.zeros(len(phys_params))
    for idx,qo in enumerate(qubit_ops):
        mean,std = qo.evaluate_with_result(final_results,
            statevector_mode=inst_bigshots.is_statevector,
            circuit_name_prefix=str(idx))
        BO_energies[idx] = np.real(mean)
        BO_energies_std[idx] = np.real(std)

    # (optionally) pad number of params for xyzpy consistency
    if not xyzpy_max_nb_params is None:
        tmp = np.zeros((len(phys_params),xyzpy_max_nb_params))
        tmp[:,:ansatz.nb_params] = Xfinal
        Xfinal = tmp

    # (optionally) dump complete results set
    dump_filename = None
    if dump_results:

        # make directory if needed
        import os
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # create hash from the time, likely to be unique
        dump_filename = results_directory+'/'+str(hash(time.time()))[:10]+'.json'
        with open(dump_filename,'w+') as dump_file:
            json.dump(accumulated_results,dump_file)

    return exact_energies+shifts,BO_energies+shifts,BO_energies_std,Xfinal,dump_filename

def _make_qubit_ops(phys_params,qubit_op_func):
    """
    Make qubit ops for `run_BO_vqe_parallel`, need to ensure all the phys_params have the
    same set of (differently weighted) Pauli operators. Also get the exact ground state 
    energies here.
    
    Parameters
    ----------
    phys_params : array
        The set of nuclear separations

    Returns
    -------
    qubit_ops : array of WeightedPauliOperator objs, size=len(phys_params)
        The set of WeightedPauliOperator corresponding to each distance
    exact_energies : array, size=len(phys_params)
        UNSHIFTED true energies of the molecular ground state at each distance
    shifts : array, size=len(phys_params)
        Molecular ground state shifts at each distance
    """
    qubit_ops = []
    exact_energies = np.zeros(len(phys_params))
    shifts = np.zeros(len(phys_params))
    for idx,pp in enumerate(phys_params):
        qubitOp,num_particles,num_spin_orbitals,shift = qubit_op_func(pp)
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
