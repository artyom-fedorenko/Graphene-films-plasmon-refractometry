import pytest
import numpy as np
from reflect import reflect_n, reflect3, reflect4, reflectance_model
from non_polarized_VV import ref_tr_vv



def test_1_boarder():
    assert abs(reflect_n(1, np.array([]), np.array([23, 23]), np.array([-23, -23]), 22))<1e-5
    assert abs(reflect_n(0, np.array([]), np.array([1, 100000]), np.array([0, 0]), 22)-1)<1e-1

def test_2_layer():
    eps_re = np.array([2.34, 1, -17])
    eps_im = np.array([-0.0061, 0, 7])
    d = np.array([1e-4])
    teta = 1
    assert np.abs(reflect_n(teta, d, eps_re, eps_im) - reflect3(d, teta, 2.34, -0.0061, 1, 0, -17, 7 )) < 1e-4

def test_3_layer():
    nu = np.linspace(100, 700, 601)
    wls = 0.01/nu
    eps_re = np.array([2.34, 1, -15, -17])
    eps_im = np.array([-0.0061, 0, 700, 7])
    thk = np.array([1e-4, 35e-9])
    teta = np.pi/4
    reflect_n_values = np.array([reflect_n(teta, thk, eps_re, eps_im,  wl) for wl in wls])
    reflect4_values = np.array([reflect4(thk[:1], thk[1], eps_re[2], eps_im[2], eps_re[3], eps_im[3], teta, wl) for wl in wls])
    assert np.all(np.abs(reflect_n_values- reflect4_values) < 1e-5)


"""def test_3_layer():
    eps_re = np.array([2.34, 1, -17])
    eps_im = np.array([-0.0061, 0, 7])
    d = np.array([1e-4])
    teta = 1
    assert np.abs(reflect_n(teta, d, eps_re, eps_im) - reflect4(d_2, d_3, teta, eps3_re, eps3_im, eps4_re, eps4_im)) < 1e-4"""

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





# teta = o: s-polar = p-polar
def test_normal_inc():
    eps_re = np.array([2.34, 1, -17])
    eps_im = np.array([-0.0061, 0, 7])
    d = np.array([1e-4])
    teta = 0
    assert np.abs(reflect_n(teta, d, eps_re, eps_im, polarization='s')-reflect_n(teta, d, eps_re, eps_im, polarization='p'))<1e-4

"""def test_bruster():
    eps_re = np.array([2.34, 1, -17])
    eps_im = np.array([-0.0061, 0, 7])
    d = np.array([1e-4])
    teta = np.atan(np.sqrt((eps_re+eps_im*1j)))
    assert np.abs(reflect_n(teta, d, eps_re, eps_im) - reflect3(d, teta, 2.34, -0.0061, 1, 0, -17, 7 )) < 1e-4"""

def test_GPT():
    nu = np.linspace(100, 700, 601)
    n1_real = 8.910
    n1_imag = -11.622
    thk = np.array([35e-9, 40.5e-6])
    eps_re = np.array([1, n1_real**2-n1_imag**2, 1.498, 1])
    eps_im = np.array([0, 2*n1_real*n1_imag, 14.1e-4, 0])
    print(list(zip(list(reflectance_model(nu, n1_real, n1_imag)), list(reflect_n(0, thk, eps_re, eps_im, 0.01/nu)))))
    assert  np.all(np.abs(reflectance_model(nu, n1_real, n1_imag) -  reflect_n(0, thk, eps_re, eps_im, 0.01/nu)) < 1e-5)

def test_less_1():
    nu = np.linspace(100, 700, 601)
    wls = 0.01/nu
    n1_real = 8.910
    n1_imag = -11.622
    eps1_re = n1_real**2 - n1_imag**2
    eps1_im = 2*n1_real*n1_imag
    thk = np.array([35e-9, 40.5e-6])
    eps_re = np.array([1, eps1_re, 1.498, 1])
    eps_im = np.array([0, eps1_im, 14.1e-4, 0])
    reflect_values = np.array([reflect_n(0, thk, eps_re, eps_im, wl) for wl in wls])
    assert np.all((0<reflect_values) & (reflect_values<1))


"""test_vv_gpt():
    nu = np.linspace(100, 700, 601)
    n1_real = 8.910
    n1_imag = -11.622
    thk = np.array([35e-9, 40.5e-6])
    eps_re = np.array([1, n1_real**2-n1_imag**2, 1.498, 1])
    eps_im = np.array([0, 2*n1_real*n1_imag, 14.1e-4, 0])
    print(list(zip(list(reflectance_model(nu, n1_real, n1_imag)), list(reflect_n(0, thk, eps_re, eps_im, 0.01/nu)))))
    assert  np.all(np.abs(reflectance_model(nu, n1_real, n1_imag) -  ref_tr_vv()[]) < 1e-5)"""
    

    