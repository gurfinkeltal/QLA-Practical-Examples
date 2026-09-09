import pennylane
from qiskit import QuantumCircuit, QuantumRegister
from QSP import compute_angles

def construct_QSVT_circuit(
    P, U, op_ancillas=1,
):
    phi = pennylane.transform_angles(compute_angles(P), "QSP", "QSVT")
    op_reg = QuantumRegister(U.num_qubits - op_ancillas, name="op")
    op_ancilla_reg = QuantumRegister(op_ancillas, name="op_ancillas")
    qsvt_ancilla = QuantumRegister(1, name="qsvt_ancilla")

    qc = QuantumCircuit(op_reg, op_ancilla_reg, qsvt_ancilla)

    def projector_rotation(angle):
        """Apply RZ(angle) when all operator ancillas are in |0>."""
        qc.x(op_ancilla_reg)
        qc.mcx(op_ancilla_reg[:], qsvt_ancilla[0])
        qc.rz(angle, qsvt_ancilla[0])
        qc.mcx(op_ancilla_reg[:], qsvt_ancilla[0])
        qc.x(op_ancilla_reg)

    U_dg = U.inverse()
    U_dg.label = U.label + "*"

    qc.h(qsvt_ancilla[0])

    for i in range(int(len(phi)/2 - 1)):
        # Projector to block-encoding and rotation. Note we require double the angle of a regular RZ gate
        projector_rotation(phi[2*i] * 2)

        # Applies U_sin
        qc.append(U, op_reg[:] + op_ancilla_reg[:])

        projector_rotation(phi[2*i + 1] * 2)

        qc.append(U_dg, op_reg[:] + op_ancilla_reg[:])

    projector_rotation(phi[-2] * 2)

    # Final block for odd polynomial
    qc.append(U, op_reg[:] + op_ancilla_reg[:])
        
    projector_rotation(phi[-1] * 2)

    qc.h(qsvt_ancilla[0])

    return qc
