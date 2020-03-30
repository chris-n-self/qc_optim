
__all__ = [
    'get_H2_qubit_op',
    'get_LiH_qubit_op',
    'run_BO_vqe',
]

import GPyOpt
import numpy as np

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries

from qiskit import Aer, execute, transpile, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator as wpo
from qiskit.aqua.algorithms import ExactEigensolver

from .cost import *
from .ansatz import *
from . import utilities as ut

def get_H2_qubit_op(dist):
    """ """

    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                         charge=0, spin=0, basis='sto3g')
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

    driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                         charge=0, spin=0, basis='sto3g')
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
               get_qubit_op,
               ansatz_type='xyz',
               seed=None,
               nb_shots=1024,
               nb_iter=30,
               nb_init='max',
               init_jobs=1,
               backend_name='qasm_simulator',
               initial_layout=None,
               optimization_level=3,
               verbose=False,
              ):
    """ """
    
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
        _nb_init = 4*900//cost_fewshots.num_circuits
    else:
        _nb_init = nb_init
    x_init = 2*np.pi*np.random.random(_nb_init*ansatz.nb_params).reshape((_nb_init,ansatz.nb_params))
    y_init = cost_fewshots(x_init)
    
    # optim
    Bopt = GPyOpt.methods.BayesianOptimization(cost_bo, 
                                               X=x_init, 
                                               Y=y_init, 
                                               **bo_args)
    Bopt.run_optimization(max_iter=nb_iter, eps=0)

    # Results found
    (x_seen, y_seen), (x_exp,y_exp) = Bopt.get_best()
    if verbose:
        print("best seen: "+f'{cost_bigshots(x_seen)}')
        print("best expected: "+f'{cost_bigshots(x_exp)}')
        print(Bopt.model.model)
        Bopt.plot_convergence()
    print(f'{np.real(cost_bigshots(x_exp)+shift)[0]}'+' '+f'{exact_energy+shift}')
    
    return np.real(cost_bigshots(x_exp)+shift)[0],exact_energy+shift
