"""
Variational Quantum Circuit that produces P_tunnel.

Circuit design (4 qubits):

    q0 --RY(f0)--*--------------RY(w)--RZ(w)--<Z>
    q1 --RY(f1)--X--*-----------RY(w)--RZ(w)--
    q2 --RY(f2)-----X--*--------RY(w)--RZ(w)--
    q3 --RY(f3)--------X--*-----RY(w)--RZ(w)--
                          |
                (ring closes q3 -> q0)

  1. ENCODING    : 4 barrier-related features -> RY rotation angles in [0, pi]
  2. ENTANGLING  : ring of CNOTs -- this is what makes the features interact
                   JOINTLY rather than independently, and is the structural
                   reason to use a circuit at all instead of a formula
  3. VARIATIONAL : trainable RY/RZ rotations (weights w), repeated n_layers times
  4. MEASUREMENT : <Z> on qubit 0 -> mapped to p = (1 - <Z>) / 2 in [0, 1]

Gradients flow via PennyLane's torch interface. Uses the `lightning.qubit`
backend (~3x faster than default.qubit for this circuit size) with adjoint
gradients.

HONESTY NOTE (must carry into the paper): the circuit is SIMULATED classically.
No quantum hardware is used and no quantum speedup is claimed.
"""
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4

try:
    _dev = qml.device("lightning.qubit", wires=N_QUBITS)
    _DIFF_METHOD = "adjoint"
except Exception:
    # fall back if lightning.qubit isn't installed in this environment
    _dev = qml.device("default.qubit", wires=N_QUBITS)
    _DIFF_METHOD = "backprop"


@qml.qnode(_dev, interface="torch", diff_method=_DIFF_METHOD)
def tunnel_circuit(features, weights):
    n_layers = weights.shape[0]
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)
    for l in range(n_layers):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)
            qml.RZ(weights[l][i][1], wires=i)
    return qml.expval(qml.PauliZ(0))


class QuantumTunnelProbability(nn.Module):
    def __init__(self, seed=0, n_layers=2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, N_QUBITS, 2, generator=g))

    def forward(self, features):
        z = tunnel_circuit(features, self.weights)
        return (1.0 - z) / 2.0

    def circuit_drawing(self, features):
        """Text diagram of the circuit -- used by the web app / presentation
        layer's quantum circuit visualizer."""
        return qml.draw(tunnel_circuit)(features, self.weights)
