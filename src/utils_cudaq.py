import cudaq
from cudaq import spin as spin_op
import openfermion as of
from openfermion.hamiltonians import s_squared_operator


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


def buildOperatorMatrix(name: str, n_qubits):
    """The name can either be: number, alpha, beta, projected or total."""

    operator = of.FermionOperator()

    if name == "number":
        for i in range(n_qubits):
            operator += of.FermionOperator(
                '{index}^ {index}'.format(index=i, ))

    elif name == "alpha":
        for i in range(n_qubits):
            if i % 2 == 0:
                operator += of.FermionOperator(
                    '{index}^ {index}'.format(index=i, ))

    elif name == "beta":
        for i in range(n_qubits):
            if i % 2 == 1:
                operator += of.FermionOperator(
                    '{index}^ {index}'.format(index=i, ))

    elif name == "projected":
        alpha_number_operator = of.FermionOperator()
        beta_number_operator = of.FermionOperator()

        for i in range(n_qubits):
            if i % 2 == 0:
                alpha_number_operator += of.FermionOperator(
                    '{index}^ {index}'.format(index=i, ))
            elif i % 2 == 1:
                beta_number_operator += of.FermionOperator(
                    '{index}^ {index}'.format(index=i, ))
        operator = 1 / 2 * (alpha_number_operator - beta_number_operator)

    elif name == "total":
        operator = s_squared_operator(n_spatial_orbitals=n_qubits // 2)

    sparse_operator = of.get_sparse_operator(operator, n_qubits=n_qubits)

    return sparse_operator