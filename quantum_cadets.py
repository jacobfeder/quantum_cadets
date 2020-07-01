import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from qiskit import *
from qiskit.circuit import ControlledGate, Gate, Instruction, Qubit, QuantumRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.circuit.library.standard_gates import C4XGate, TGate, XGate, RZGate, RXGate
from qiskit.ignis.characterization.characterization_utils import pad_id_gates
from qiskit.circuit.library.generalized_gates import MCMT
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import phase_damping_error
from typing import Union, Callable, List, Tuple

def encode_X(num, qubits, register_size):
	non_register = qubits - register_size
	Xs = np.array([int(ch) for ch in np.binary_repr(num, width=register_size)], dtype=bool)
	registers = np.arange(register_size)
	return registers[Xs] + non_register, registers[np.invert(Xs)] +non_register

def build_encode_circuit(num, qubits, register_size):
	""" Create the registery conversion circuit. Assume the last qubits are the register.
	qubits [int]: Total number of qubits in the global circuit
	register_size [int]: Total number of qubits allocated for the register.
	num [int]: target encoding
	"""

	# generate the X-gate configuration
	CGates, XGates = encode_X(num, qubits, register_size)
	# create a quantum circuit acting on the registers
	conv_register = MCMT(XGate(), len(CGates), len(XGates))
	XRange = [*CGates, *XGates]

	return conv_register, XRange

def quantum_cadets(n_qubits, noise_circuit, damping_error=0.02):
	"""
	n_qubits [int]: total number of qubits - 2^(n_qubits-1) is the number of qft points, plus one sensing qubit
	noise_circuit [function]: function that takes one input (a time index between 0-1) and returns a quantum circuit with 1 qubit
	damping_error [float]: T2 damping error
	"""
	register_size = n_qubits - 1

	# Create a Quantum Circuit acting on the q register
	qr = QuantumRegister(n_qubits, 'q')
	cr = ClassicalRegister(register_size)
	qc = QuantumCircuit(qr, cr)

	# Add a H gate on qubit 1,2,3...N-1
	for i in range(register_size):
		qc.h(i+1)

	# multi-qubit controlled-not (mcmt) gate
	mcmt_gate = MCMT(XGate(), register_size, 1)
	qr_range=[*range(1, n_qubits), 0]

	for bit in range(2**register_size):
		qc.append(mcmt_gate, [qr[i] for i in qr_range])
		# external noise gates
		qc.append(noise_circuit(bit / 2**register_size), [qr[0]])
		qc.append(mcmt_gate, [qr[i] for i in qr_range])

		if bit == 0:
			for i in range(register_size):
				qc.x(i + (n_qubits - register_size))
		elif bit == 2**register_size - 1:
			pass
		else:
			conv_register, XRange = build_encode_circuit(bit, n_qubits, register_size)
			qc.append(conv_register, qr[XRange])

	# run the QFT
	qft = circuit.library.QFT(register_size)
	qc.append(qft, qr[1:n_qubits])

	# map the quantum measurement to classical bits
	qc.measure(range(1, n_qubits), range(0, register_size))

	# display the quantum circuit in text form
	print(qc.draw('text'))
	# qc.draw('mpl')
	plt.show()

	# noise model
	t2_noise_model = NoiseModel()
	t2_noise_model.add_quantum_error(phase_damping_error(damping_error), 'id', [0])

	# run the quantum circuit on the statevector simulator backend
	#backend = Aer.get_backend('statevector_simulator')
	# run the quantum circuit on the qasm simulator backend
	backend = Aer.get_backend('qasm_simulator')
	# number of histogram samples
	shots = 10000

	# execute the quantum program
	job = execute(qc, backend, noise_model=t2_noise_model, shots=shots)
	# outputstate = result.get_statevector(qc, decimals=3)
	# visualization.plot_state_city(outputstate)
	result = job.result()
	# collect the state histogram counts
	counts = result.get_counts(qc)
	#plot_histogram(counts)

	qft_result = np.zeros(2**register_size)
	for f in range(len(qft_result)):
		# invert qubit order and convert to string
		f_bin_str = ('{0:0' + str(register_size) + 'b}').format(f)[::-1]
		if f_bin_str in counts:
			if f:
				# flip frequency axis and assign histogram counts
				qft_result[2**register_size - f] = counts[f_bin_str] / shots
			else:
				# assign histogram counts, no flipping because of qft representation (due to nyquist sampling?)
				qft_result[0] = counts[f_bin_str] / shots

	freq = np.arange(2**register_size)
	plt.scatter(freq, qft_result, label='QFT')
	plt.xlabel('Frequency (Hz)')

	# add interpolation to make the plot look nicer
	interp_deg = 10
	interp_freq = np.arange(interp_deg*2**register_size) / interp_deg
	interp_qft_result = signal.resample(qft_result, interp_deg*2**register_size)
	plt.plot(interp_freq, interp_qft_result, label='QFT (interpolation)')
	
	# print the final measurement results
	print('QFT spectrum:') 
	print(qft_result)

	# show the plots
	plt.show()

if __name__ == "__main__":
	def narrowband_noise(time, f):
		"""
		Apply a single-frequency noise source
		"""
		qr = QuantumRegister(1)
		qc = QuantumCircuit(qr, name='narrowband_noise')
		qc.append(RZGate(2 * np.pi * f * time), [qr[0]])
		return qc

	def broadband_noise(time):
		"""
		Apply a large number of identity gates, which will
		accumulate errors due to the inherent T2 noise
		"""
		n_ids = 40
		qr = QuantumRegister(1)
		qc = QuantumCircuit(qr, name='broadband_noise')
		qc = pad_id_gates(qc, qr, 0, n_ids)		
		return qc

	def combo(time):
		"""
		Combine the narrowband and broadband noise circuits
		"""
		qr = QuantumRegister(1)
		qc = QuantumCircuit(qr, name='combo_noise')
		qc.append(narrowband_noise(time, 3), [qr[0]])
		qc.append(broadband_noise(time), [qr[0]])
		return qc

	# arg 1: number of qubits (QFT size)
	# arg 2: noise function
	quantum_cadets(4, combo, damping_error=0.02)
