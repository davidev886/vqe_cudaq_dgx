import numpy as np

import cudaq
from src.utils_cudaq import buildOperatorMatrix
import pandas as pd
from scipy.optimize import minimize
import cma

class VqeQnp(object):
    def __init__(self,
                 n_qubits,
                 n_layers,
                 init_mo_occ=None,
                 target="nvidia",
                 system_name="FeNTA"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.number_of_Q_blocks = n_qubits // 2 - 1
        self.num_params = 2 * self.number_of_Q_blocks * n_layers
        self.init_mo_occ = init_mo_occ
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.target = target
        self.initial_x_gates_pos = self.prepare_initial_circuit()

        self.spin_s_square = buildOperatorMatrix("total", n_qubits)
        self.spin_s_z = buildOperatorMatrix("projected", n_qubits)
        self.system_name = system_name
        self.num_qpus = 0

    def prepare_initial_circuit(self):
        """
        Creates a list with the position of the X gates that should be applied to the initial |00...0>
        state to set the number of electrons and the spin correctly
        """
        x_gates_pos_list = []
        if self.init_mo_occ is not None:
            for idx_occ, occ in enumerate(self.init_mo_occ):
                if int(occ) == 2:
                    x_gates_pos_list.extend([2 * idx_occ, 2 * idx_occ + 1])
                elif int(occ) == 1:
                    x_gates_pos_list.append(2 * idx_occ)

        return x_gates_pos_list

    def layers(self):
        """
            Generates the QNP ansatz circuit and returns the  kernel and the optimization paramenters thetas

            params: list/np.array
            [theta_0, ..., theta_{M-1}, phi_0, ..., phi_{M-1}]
            where M is the total number of blocks = layer * (n_qubits/2 - 1)

            returns: kernel
                     thetas
        """
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        number_of_blocks = self.number_of_Q_blocks
        # cudaq.set_target("nvidia-mgpu") # nvidia or nvidia-mgpu or tensornet-mps
        if self.target != "":
            cudaq.set_target(self.target)  # nvidia or nvidia-mgpu
            target = cudaq.get_target()
            self.num_qpus = target.num_qpus()
            print("num_gppus=", target.num_qpus())
        else:
            self.num_qpus = 0

        kernel, thetas = cudaq.make_kernel(list)
        # Allocate n qubits.
        qubits = kernel.qalloc(n_qubits)

        for init_gate_position in self.initial_x_gates_pos:
            kernel.x(qubits[init_gate_position])
        # if self.num_qpus > 1:
        #     spin_value_initial = cudaq.observe(kernel,
        #                                        self.spin_s_square,
        #                                        [],
        #                                        execution=cudaq.parallel.thread
        #                                        ).expectation()
        #
        #     spin_proj_initial = cudaq.observe(kernel,
        #                                       self.spin_s_z,
        #                                       [],
        #                                       execution=cudaq.parallel.thread
        #                                       ).expectation()
        # else:
        #     spin_value_initial = cudaq.observe(kernel, self.spin_s_square, []).expectation()
        #     spin_proj_initial = cudaq.observe(kernel, self.spin_s_z, []).expectation()
        # print("initial S^2:", spin_value_initial)
        # print("initial S_z:", spin_proj_initial)

        count_params = 0
        for idx_layer in range(n_layers):
            for starting_block_num in [0, 1]:
                for idx_block in range(starting_block_num, number_of_blocks, 2):
                    qubit_list = [qubits[2 * idx_block + j] for j in range(4)]
                    # print(idx_block,
                    #      "theta",
                    #      idx_layer * number_of_blocks + idx_block,
                    #      [2 * idx_block + j for j in range(4)]
                    #      )

                    # PX gates decomposed in terms of standard gates
                    # and NO controlled Y rotations.
                    # See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010

                    a, b, c, d = qubit_list
                    kernel.cx(d, b)
                    kernel.cx(d, a)
                    kernel.rz(parameter=-np.pi / 2, target=a)
                    kernel.s(b)
                    kernel.h(d)
                    kernel.cx(d, c)
                    kernel.cx(b, a)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.rz(parameter=+np.pi / 2, target=a)
                    kernel.cz(a, d)
                    kernel.cx(a, c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=c)
                    kernel.cx(b, c)
                    kernel.cx(b, d)
                    kernel.rz(parameter=+np.pi / 2, target=b)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=d)
                    kernel.cx(a, c)
                    kernel.cz(a, d)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=d)
                    kernel.cx(d, c)
                    kernel.h(d)
                    kernel.cx(d, b)
                    kernel.s(d)
                    kernel.rz(parameter=-np.pi / 2, target=b)
                    kernel.cx(b, a)
                    count_params += 1

                    # Orbital rotation
                    kernel.fermionic_swap(np.pi, b, c)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], a, b)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], c, d)
                    kernel.fermionic_swap(np.pi, b, c)
                    count_params += 1

        return kernel, thetas

    def get_state_vector(self, param_list):
        """
        Returns the state vector generated by the ansatz with paramters given by param_list
        """
        kernel, thetas = self.layers()
        state = np.array(cudaq.get_state(kernel, param_list), dtype=complex)
        return state

    def run_vqe_cudaq(self, hamiltonian, options=None):
        """
        Run VQE
        """
        optimizer = cudaq.optimizers.COBYLA()
        #optimizer = cudaq.optimizers.LBFGS()
        optimizer.initial_parameters = np.random.rand(self.num_params)
        kernel, thetas = self.layers()
        maxiter = options.get('maxiter', 100)
        optimizer.max_iterations = options.get('maxiter', maxiter)
        optimizer_type = options.get('optimizer_type', "cudaq")
        # optimizer...

        exp_vals = []

        def eval(theta):
            print("inside eval")
            if self.num_qpus > 1:
                exp_val = cudaq.observe(kernel,
                                        hamiltonian,
                                        theta,
                                        execution=cudaq.parallel.thread).expectation()
            else:
                print("inside eval num_qpus == 1")
                exp_val = cudaq.observe(kernel,
                                    hamiltonian,
                                    theta).expectation()
                print("inside eval ->", exp_val)
            exp_vals.append(exp_val)
            if isinstance(optimizer, cudaq.optimizers.LBFGS):
                d_1 = 1 / 2.
                d_2 = (np.sqrt(2) - 1) / 4.
                alpha = np.pi / 2
                beta = np.pi

                gradient_list = [0] * len(theta)

                for j in range(len(theta)):
                    new_theta = theta[:]
                    new_theta[j] = theta[j] + alpha
                    if self.num_qpus > 1:
                        term_1 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] - alpha
                        term_2 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] + beta
                        term_3 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] - beta
                        term_4 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()
                    else:
                        term_1 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] - alpha
                        term_2 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] + beta
                        term_3 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] - beta
                        term_4 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                    gradient_list[j] = d_1 * (term_1 - term_2) - d_2 * (term_3 - term_4)

                return exp_val, gradient_list
            else:

                return exp_val

        # def callback_func(theta):
        #     exp_val = cudaq.observe(kernel, hamiltonian, theta).expectation()
        #     exp_vals.append(exp_val)

        def to_minimize(theta):
            if self.num_qpus > 1:
                exp_val = cudaq.observe(kernel,
                                        hamiltonian,
                                        theta,
                                        execution=cudaq.parallel.thread).expectation()
            else:
                exp_val = cudaq.observe(kernel,
                                        hamiltonian,
                                        theta).expectation()

            exp_vals.append(exp_val)
            return exp_val

        def compute_gradient(theta):
                d_1 = 1 / 2.
                d_2 = (np.sqrt(2) - 1) / 4.
                alpha = np.pi / 2
                beta = np.pi

                gradient_list = [0] * len(theta)

                for j in range(len(theta)):
                    new_theta = theta[:]
                    new_theta[j] = theta[j] + alpha
                    if self.num_qpus > 1:
                        term_1 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] - alpha
                        term_2 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] + beta
                        term_3 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()

                        new_theta[j] = theta[j] - beta
                        term_4 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta,
                                               execution=cudaq.parallel.thread).expectation()
                    else:
                        term_1 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] - alpha
                        term_2 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] + beta
                        term_3 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                        new_theta[j] = theta[j] - beta
                        term_4 = cudaq.observe(kernel,
                                               hamiltonian,
                                               new_theta).expectation()

                    gradient_list[j] = d_1 * (term_1 - term_2) - d_2 * (term_3 - term_4)

                return gradient_list

        if optimizer_type == "cudaq":
            print("Using cudaq optimizer")
            energy, parameter = optimizer.optimize(self.num_params, eval)
        elif optimizer_type == "scipy":
            print("Using scipy optimizer")
            x0 = np.random.uniform(low=0, high=2 * np.pi, size=self.num_params)
            result = minimize(to_minimize,
                              x0,
                              jac=compute_gradient,
                              method='L-BFGS-B',
                              # callback=callback_func,
                              options={'maxiter': maxiter,
                                       'disp': True,
                                       })
            parameter = result.x
            energy = result.fun
        elif optimizer_type == "cma":
            print("cma optimizer")
            sigma = 1

            initial_parameters = options.get("initial_parameters", None)
            if initial_parameters is None:
                x0 = np.random.uniform(low=-np.pi, high=np.pi, size=self.num_params)
                print(f'starting training with random parameters')
                print(x0)
            else:
                x0 = np.pad(initial_parameters,
                            (0, self.num_params - len(initial_parameters)),
                            constant_values=0.01)
                print(f'starting training with previous parameters')
                print(x0)

            print(f'starting training with sigma at value {sigma}')

            bounds = [self.num_params * [-np.pi], self.num_params * [np.pi]]
            options_opt = {'bounds': bounds,
                           'maxfevals': maxiter,
                           'verbose': -3,
                           'tolfun': 1e-5}
            es = cma.CMAEvolutionStrategy(x0, sigma, options_opt)
            es.optimize(to_minimize)
            res = es.result
            energy = res.fbest
            parameter = res.xbest

        info_final_state = dict()
        print("")
        print("Num Params:", self.num_params)
        print("qubits:", self.n_qubits)
        print("n_layers:", self.n_layers)
        print("Energy after the VQE:", energy)

        if self.num_qpus > 1:
            spin_value = cudaq.observe(kernel,
                                               self.spin_s_square,
                                               parameter,
                                               execution=cudaq.parallel.thread
                                               ).expectation()

            spin_proj = cudaq.observe(kernel,
                                              self.spin_s_z,
                                              parameter,
                                              execution=cudaq.parallel.thread
                                              ).expectation()
        else:
            spin_value = cudaq.observe(kernel, self.spin_s_square, parameter).expectation()
            spin_proj = cudaq.observe(kernel, self.spin_s_z, parameter).expectation()

        print("S^2:", spin_value)
        print("S_z:", spin_proj)
        info_final_state["S^2"] = spin_value
        info_final_state["S_z"] = spin_proj
        info_final_state["energy_optimized"] = energy

        df = pd.DataFrame(info_final_state, index=[0])
        df.to_csv(f'{self.system_name}_info_final_state_{self.n_layers}_layers_opt_{optimizer_type}.csv', index=False)
        return energy, parameter, exp_vals

    def compute_energy(self, hamiltonian, params):
        kernel, thetas = self.layers()

        if self.num_qpus > 1:
            exp_val = cudaq.observe(kernel,
                                    hamiltonian,
                                    params,
                                    execution=cudaq.parallel.thread).expectation()
        else:
            exp_val = cudaq.observe(kernel,
                                    hamiltonian,
                                    params).expectation()
        print("parameters", params)
        print("energy from vqe", exp_val)
        return exp_val