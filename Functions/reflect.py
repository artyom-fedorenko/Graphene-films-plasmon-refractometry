import numpy as np
import matplotlib.pyplot as plt





# define reflect3 by V.V.'s code, but implement globals in function
# d_2 - air gap, teta - internal incidence angle, eps_3 - permittivity of 3rd medium (InSb)
def reflect3(d_2, teta, eps1_re, eps1_im, eps2_re, eps2_im,  eps3_re, eps3_im, wavelength = 197e-6):

    n_1 = 1.531 - 1j * 0.002  # Complex refractive index of the prism (Zeonex)
    eps_1 = eps1_re + eps1_im*1j  # Dielectric permittivity of the prism (Zeonex)       
    eps_2 = eps2_re + eps2_im*1j  # Dielectric permittivity of air
    eps_3 = eps3_re + eps3_im*1j
    
    # Normal components of wave vectors (normalized by k0)
    kz_1 = np.sqrt(eps_1) * np.cos(teta)
    kz_2 = np.sqrt(eps_2 - eps_1 * np.sin(teta)**2)
    kz_3 = np.sqrt(eps_3 - eps_1 * np.sin(teta)**2)

    # Reflection coefficients
    r_12 = (eps_2 * kz_1 - eps_1 * kz_2) / (eps_2 * kz_1 + eps_1 * kz_2)
    r_23 = (eps_3 * kz_2 - eps_2 * kz_3) / (eps_3 * kz_2 + eps_2 * kz_3)

    # Transmission coefficients
    t_12 = 2 * kz_1 * np.sqrt(eps_1 * eps_2) / (eps_2 * kz_1 + eps_1 * kz_2)
    t_23 = 2 * kz_2 * np.sqrt(eps_2 * eps_3) / (eps_3 * kz_2 + eps_2 * kz_3)

    # Transformation matrix
    S1 = np.array([[1 / t_12, r_12 / t_12], [r_12 / t_12, 1 / t_12]])

    R = np.zeros(len(d_2))  # Initialize reflection coefficient array

    print(f'kz{[kz_1, kz_2, kz_3]}\n r{[r_12, r_23]}\n t{[t_12, t_23]}\nRfr1 {S1}\n')


    for j in range(len(d_2)):
        S2 = np.array([[np.exp(-1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23,
                        r_23 * np.exp(-1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23],
                    [r_23 * np.exp(1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23,
                        np.exp(1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23]], dtype=np.complex128)
        
        print(f'S2{S2}')
        
        S = np.dot(S1, S2)
        print(f'S{S}')
        R[j] = (np.abs(S[1, 0] / S[0, 0]))**2

    return R



# define reflect4 by V.V.'s code

def reflect4(d_2, d_3, teta, eps3_re, eps3_im, eps4_re, eps4_im, wavelength = 197 * 10**(-6)): # teta - internal incidence angle,  d_2 - gap, d_3 - Gr, eps_re/im - gr
    
    n_1 = 1.531 - 1j * 0.002  # Complex refractive index of the prism (Zeonex)
    eps_1 = n_1**2  # Dielectric permittivity of the prism (Zeonex)
    eps_2 = 1  #Dielectric permittivity of air"
    eps_3 = eps3_re+eps3_im*1j
    eps_4 = eps4_re+eps4_im*1j

    # Normal components of wave vectors (normalized by k0)
    kz_1 = np.sqrt(eps_1) * np.cos(teta)
    kz_2 = np.sqrt(eps_2 - eps_1 * np.sin(teta)**2)
    kz_3 = np.sqrt(eps_3 - eps_1 * np.sin(teta)**2)
    kz_4 = np.sqrt(eps_4 - eps_1 * np.sin(teta)**2)

    # Reflection coefficients
    r_12 = (eps_2 * kz_1 - eps_1 * kz_2) / (eps_2 * kz_1 + eps_1 * kz_2)
    r_23 = (eps_3 * kz_2 - eps_2 * kz_3) / (eps_3 * kz_2 + eps_2 * kz_3)
    r_34 = (eps_4 * kz_3 - eps_3 * kz_4) / (eps_4 * kz_3 + eps_3 * kz_4)

    # Transmission coefficients
    t_12 = 2 * kz_1 * np.sqrt(eps_1 * eps_2) / (eps_2 * kz_1 + eps_1 * kz_2)
    t_23 = 2 * kz_2 * np.sqrt(eps_2 * eps_3) / (eps_3 * kz_2 + eps_2 * kz_3)
    t_34 = 2 * kz_3 * np.sqrt(eps_3 * eps_4) / (eps_4 * kz_3 + eps_3 * kz_4)

    # Transformation matrices
    S1 = np.array([[1 / t_12, r_12 / t_12], [r_12 / t_12, 1 / t_12]])
    S3 = np.array([[np.exp(-1j * (2 * np.pi / wavelength) * kz_3 * d_3) / t_34,
                    r_34 * np.exp(-1j * (2 * np.pi / wavelength) * kz_3 * d_3) / t_34],
                    [r_34 * np.exp(1j * (2 * np.pi / wavelength) * kz_3 * d_3) / t_34,
                    np.exp(1j * (2 * np.pi / wavelength) * kz_3 * d_3) / t_34]])

    R = np.zeros(len(d_2))  # Initialize reflection coefficient array

    for j in range(len(d_2)):
        S2 = np.array([[np.exp(-1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23,
                        r_23 * np.exp(-1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23],
                        [r_23 * np.exp(1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23,
                        np.exp(1j * (2 * np.pi / wavelength) * kz_2 * d_2[j]) / t_23]], dtype=np.complex128)
        
        S = S1 @ S2 @ S3  # Matrix multiplication
        R[j] = (np.abs(S[1, 0] / S[0, 0]))**2
    
    return R


def reflect_n(number:int, teta: float, thickness: np.ndarray, eps_re: np.ndarray, eps_im: np.ndarray, wavelength = 197*1e-6):
    
    assert len(thickness)+2 == len(eps_re) == len(eps_im), 'length does not match'
    

    
    eps = eps_re + eps_im*1j

    # wave vectors
    kz = np.zeros(number, dtype=complex)
    kz[0] = np.sqrt(eps[0]) * np.cos(teta)

    for i in range(number-1):
        kz[i+1] = np.sqrt(eps[i+1] - eps[0] * np.sin(teta)**2)


    # Fresnel coefficients
    r = np.zeros(number-1, dtype=complex)
    t = np.zeros(number-1, dtype=complex)

    for i in range(len(r)):
        r[i] = (eps[i+1] * kz[i] - eps[i] * kz[i+1]) / (eps[i+1] * kz[i] + eps[i] * kz[i+1])

    for i in range(len(t)):
        t[i] = 2 * kz[i] * np.sqrt(eps[i] * eps[i+1]) / (eps[i+1] * kz[i] + eps[i] * kz[i+1])


    
    # Propagation matrix
    Pr = np.zeros((number-2, 2, 2), dtype=complex)

    for i in range(Pr.shape[0]):
        phase = 1j * (2 * np.pi ) * kz[i+1] * thickness[i] / wavelength
        print(phase)
        Pr[i] = np.diag([np.exp(-phase), np.exp(phase)])


    # Refraction matrix
    Rfr = np.zeros((number-1, 2, 2), dtype=complex)

    for i in range(Rfr.shape[0]):
        Rfr[i] = np.array([[1 / t[i], r[i] / t[i]], [r[i] / t[i], 1 / t[i]]])


    S = Rfr[0]

    print(f'kz{kz}\n r{r}\n t{t}\n Pr{Pr}\n Rfr{Rfr}\n')

    for i in range(number-2):
        S = S@Pr[i]@Rfr[i+1]
        print(f'S2{Pr[i]@Rfr[i+1]}')
        print(f'S{S}')

    
    # reflection coefficient
    R = (np.abs(S[1, 0] / S[0, 0]))**2
    return R




eps_re = np.array([2.34, 1, -17])
eps_im = np.array([-0.0061, 0, 7])
d = np.array([8*1e-5])

teta = 0.736

R1 = reflect_n(3, teta, d, eps_re, eps_im)
R2 = reflect3(d, teta, 2.34, -0.0061, 1, 0, -17, 7 )

print(R1, R2)