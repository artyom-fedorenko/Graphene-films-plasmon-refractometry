import pytest
import numpy as np
from reflect import reflect_n, reflect3

def test_2_layer():
    eps_re = np.array([2.34, 1, -17])
    eps_im = np.array([-0.0061, 0, 7])
    d = np.array([1e-4])
    teta = 1
    assert np.abs(reflect_n(teta, d, eps_re, eps_im) - reflect3(d, teta, 2.34, -0.0061, 1, 0, -17, 7 )) < 1e-4

def test_matrix_overflow():
    with pytest.raises(ValueError) as e_info:
        eps_re = np.array([2.34, 1, -17])
        eps_im = np.array([-0.0061, 0, 7])
        d = np.array([10])
        teta = 1
        result = reflect_n(teta, d, eps_re, eps_im)
    assert ("Overflow in matrix exponent, probably because of wrong units of d." in str(e_info.value))



""" pytest.raises is used to test whether a specific exception is raised during the execution of a block of code. 
        It does not automatically output the exception; rather, it captures it so you can assert its properties 
        (e.g., type, message).
"""

#  assert "Lengths of thickness, eps_re, and eps_im must match" in str(excinfo.value) - to test error message