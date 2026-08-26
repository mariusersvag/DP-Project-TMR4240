import numpy as np


def wrap_angle(angle_rad):
	"""Wrap an angle in radians to the interval (-pi, pi]."""
	wrapped = (angle_rad + np.pi) % (2.0 * np.pi) - np.pi
	if np.isclose(wrapped, -np.pi):
		wrapped = np.pi
	return float(wrapped)

def wrap_eta(eta: np.ndarray) -> np.ndarray:
	"""Wrap roll, pitch, and yaw in an eta vector to (-pi, pi]."""
	eta_wrapped = eta.copy()
	eta_wrapped[3:6] = [wrap_angle(angle) for angle in eta[3:6]]
	return eta_wrapped