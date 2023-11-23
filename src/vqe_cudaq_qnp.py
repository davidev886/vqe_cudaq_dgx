import numpy as np

import cudaq
from src.utils_cudaq import buildOperatorMatrix
import pandas as pd


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
        # cudaq.set_target("nvidia-mgpu") # nvidia or nvidia-mgpu
        cudaq.set_target(self.target)  # nvidia or nvidia-mgpu
        kernel, thetas = cudaq.make_kernel(list)
        # Allocate n qubits.
        qubits = kernel.qalloc(n_qubits)

        for init_gate_position in self.initial_x_gates_pos:
            kernel.x(qubits[init_gate_position])
        spin_value_initial = cudaq.observe(kernel, self.spin_s_square, []).expectation_z()

        spin_proj_initial = cudaq.observe(kernel, self.spin_s_z, []).expectation_z()
        print("initial S^2:", spin_value_initial)
        print("initial S_z:", spin_proj_initial)

        count_params = 0
        for idx_layer in range(n_layers):
            for starting_block_num in [0, 1]:
                for idx_block in range(starting_block_num, number_of_blocks, 2):
                    qubit_list = [qubits[2 * idx_block + j] for j in range(4)]
                    print([2 * idx_block + j for j in range(4)])
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

    def run_vqe_cudaq(self, hamiltonian, options=None):
        """
        Run VQE
        """
        # optimizer = cudaq.optimizers.NelderMead()
        optimizer = cudaq.optimizers.LBFGS()
        optimizer.initial_parameters = np.random.rand(self.num_params)
        kernel, thetas = self.layers()
        optimizer.max_iterations = options.get('maxiter', 100)

        # optimizer...

        # Finally, we can pass all of that into `cudaq.vqe` and it will automatically run our
        # optimization loop and return a tuple of the minimized eigenvalue of our `spin_operator`
        # and the list of optimal variational parameters.
        # energy, parameter = cudaq.vqe(
        #    kernel=kernel,
        #    spin_operator=hamiltonian,
        #    optimizer=optimizer,
        #    parameter_count=self.num_params)

        # def eval(theta):
        #    # Callback goes here
        #    value = cudaq.observe(kernel, hamiltonian, thetas).expectation_z()
        #    print(value)
        #    return value

        exp_vals = []

        def eval(theta):
            exp_val = cudaq.observe(kernel, hamiltonian, theta).expectation_z()

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
                    term_1 = cudaq.observe(kernel, hamiltonian, new_theta).expectation_z()

                    new_theta[j] = theta[j] - alpha
                    term_2 = cudaq.observe(kernel, hamiltonian, new_theta).expectation_z()

                    new_theta[j] = theta[j] + beta
                    term_3 = cudaq.observe(kernel, hamiltonian, new_theta).expectation_z()

                    new_theta[j] = theta[j] - beta
                    term_4 = cudaq.observe(kernel, hamiltonian, new_theta).expectation_z()

                    gradient_list[j] = d_1 * (term_1 - term_2) - d_2 * (term_3 - term_4)

                return exp_val, gradient_list
            else:

                return exp_val

        energy, parameter = optimizer.optimize(self.num_params, eval)
        # energy, parameter = optimizer.optimize(self.num_params, eval)

        info_final_state = dict()

        print("")
        print("Num Params:", self.num_params)
        print("qubits:", self.n_qubits)
        print("n_layers:", self.n_layers)
        print("Energy after the VQE:", energy)

        spin_value = cudaq.observe(kernel, self.spin_s_square, parameter).expectation_z()

        spin_proj = cudaq.observe(kernel, self.spin_s_z, parameter).expectation_z()
        print("S^2:", spin_value)
        print("S_z:", spin_proj)
        info_final_state["S^2"] = spin_value
        info_final_state["S_z"] = spin_proj
        info_final_state["energy_optimized"] = energy

        df = pd.DataFrame(info_final_state, index=[0])
        df.to_csv(f'{self.system_name}_info_final_state_{self.n_layers}_layers.csv', index=False)
        return energy, parameter, exp_vals
