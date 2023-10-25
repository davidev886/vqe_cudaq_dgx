import cudaq
from cudaq import spin as spin_op


def from_string_to_cudaq_spin(pauli_string, qubit):
    if pauli_string.lower() in ('id', 'i'):
        return 1
    elif pauli_string.lower() == 'x':
        return spin_op.x(qubit)
    elif pauli_string.lower() == 'y':
        return spin_op.y(qubit)
    elif pauli_string.lower() == 'z':
        return spin_op.z(qubit)


def get_cudaq_hamiltonian(jw_hamiltonian):
    """ Converts an openfermion QubitOperator Hamiltonian into a cudaq.SpinOperator Hamiltonian

    """

    hamiltonian_cudaq = 0.0
    for ham_term in jw_hamiltonian:
        [(operators, ham_coeff)] = ham_term.terms.items()
        if len(operators):
            cuda_operator = 1.0
            for qubit_index, pauli_op in operators:
                cuda_operator *= from_string_to_cudaq_spin(pauli_op, qubit_index)
        else:
            cuda_operator = from_string_to_cudaq_spin('id', 0)
        cuda_operator = ham_coeff * cuda_operator
        hamiltonian_cudaq += cuda_operator

    return hamiltonian_cudaq
