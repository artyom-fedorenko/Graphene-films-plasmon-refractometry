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




"""
Calculate the reflection coefficient of a layered medium using Fresnel coefficients 
and transfer matrix formalism.

Parameters:
    number (int): Total number of layers (including incident and substrate layers).
    teta (float): Angle of incidence in radians.
    thickness (np.ndarray): Array of thicknesses for internal layers (length = number-2).
    eps_re (np.ndarray): Array of real parts of permittivity (length = number).
    eps_im (np.ndarray): Array of imaginary parts of permittivity (length = number).
    wavelength (float): Wavelength of light in the same units as thickness.

Returns:
    float: Reflection coefficient (R).


Warning: S is not a scattering matrix because 1/tij was eliminated for better computation!
"""



"""GPT-o3-mini-high reflectance function for fitting Fourier"""


def reflectance_model(nu, n1_real, n1_imag):
    """
    Calculate the reflectance for a four–layer system at normal incidence:
    
      Air (n0) / Composite Graphene Film (n1, d1) / Polypropylene Film (n2, d2) / Air (n3)
    
    Parameters:
      nu       : Wavenumber array [cm^-1]
      n1_real  : Real part of the unknown refractive index of the composite graphene film
      n1_imag  : Imaginary part of the unknown refractive index of the composite graphene film
      
    Returns:
      R        : Reflectance (power reflection coefficient) as a function of nu.
    """
    # --- Known parameters ---
    # Ambient and final medium (air)
    n0 = 1.0
    n3 = 1.0
    
    # Composite graphene film (unknown optical constants)
    n1 = n1_real + 1j*n1_imag
    d1 = 35e-7      # 35 nm in cm (1 nm = 1e-7 cm)
    
    # Polypropylene film (substrate)
    n2 = 1.498 + 1j*(14.1e-4)
    d2 = 40.5e-4    # 40.5 µm in cm (1 µm = 1e-4 cm)
    
    # --- Compute the phase thickness for each layer ---
    # Note: nu is in cm^-1, and lambda = 1/nu.
    delta1 = 2 * np.pi * n1 * d1 * nu   # For layer 1 (graphene film)
    delta2 = 2 * np.pi * n2 * d2 * nu   # For layer 2 (polypropylene film)
    
    # --- Construct the 2x2 characteristic matrices for each layer ---
    # For a given layer j with refractive index n_j and phase thickness delta_j, the matrix is:
    #   M_j = [ cos(delta_j), i*sin(delta_j)/n_j ]
    #         [ i*n_j*sin(delta_j), cos(delta_j)    ]
    # 
    # Layer 1:
    M11_1 = np.cos(delta1)
    M12_1 = 1j * np.sin(delta1) / n1
    M21_1 = 1j * n1 * np.sin(delta1)
    M22_1 = np.cos(delta1)
    
    # Layer 2:
    M11_2 = np.cos(delta2)
    M12_2 = 1j * np.sin(delta2) / n2
    M21_2 = 1j * n2 * np.sin(delta2)
    M22_2 = np.cos(delta2)
    
    # --- Total characteristic matrix ---
    # For the two layers in series, the total matrix M = M1 * M2.
    # For each nu (elementwise multiplication):
    M11 = M11_1 * M11_2 + M12_1 * M21_2
    M12 = M11_1 * M12_2 + M12_1 * M22_2
    M21 = M21_1 * M11_2 + M22_1 * M21_2
    M22 = M21_1 * M12_2 + M22_1 * M22_2
    
    # --- Reflection coefficient ---
    # For a stack with total matrix M, the reflection amplitude is given by:
    #   r = [(M11 + M12*n3)*n0 - (M21 + M22*n3)] / [(M11 + M12*n3)*n0 + (M21 + M22*n3)]
    r_total = ((M11 + M12 * n3) * n0 - (M21 + M22 * n3)) / ((M11 + M12 * n3) * n0 + (M21 + M22 * n3))
    
    # Reflectance:
    R = np.abs(r_total)**2
    return R  # Return real values




"""«You aren't gonna need it»"""



def reflect_n(teta: float, thickness: np.ndarray, eps_re: np.ndarray, eps_im: np.ndarray, wavelength:float = 197*1e-6, polarization = 'p') -> float:
    

    assert len(thickness)+2 == len(eps_re) == len(eps_im), 'length does not match'

    if(teta < 0 or teta > np.pi/2):
        raise ValueError("Incident angle out of range. Incident angle must be within (0, pi/2)") 
    
    if np.any(thickness <= 0):
        raise ValueError("Thickness must be more than zero")     

    number = len(eps_re)  # number of layers
    eps = eps_re + eps_im*1j

    if (number == 2):
        kz0 = np.sqrt(eps[0]) * np.cos(teta)
        kz1 = np.sqrt(eps[1]-eps[0] * np.sin(teta)**2)

        if(polarization == 'p'):
            r = (eps[1] * kz0 - eps[0] * kz1) / (eps[1] * kz0 + eps[0] * kz1)
        if (polarization == 's'):
            r = (kz0 - kz1) / (kz0 + kz1)

        return abs(r**2)
    
    if (number < 2):
        raise ValueError ("At least 2 layers required.")

    if (number > 2):  

        # wave vectors
        kz = np.zeros(number, dtype=np.complex128)
        kz[0] = np.sqrt(eps[0]) * np.cos(teta)

        for i in range(number-1):
            kz[i+1] = np.sqrt(eps[i+1] - eps[0] * np.sin(teta)**2)


        # Fresnel coefficients
        r = np.zeros(number-1, dtype=np.complex128)
        t = np.zeros(number-1, dtype=np.complex128) # not used but may be in future

        if(polarization=='p'):    
            for i in range(number - 1):
                r[i] = (eps[i+1] * kz[i] - eps[i] * kz[i+1]) / (eps[i+1] * kz[i] + eps[i] * kz[i+1])
            for i in range(number - 1):
                t[i] = 2 * kz[i] * np.sqrt(eps[i] * eps[i+1]) / (eps[i+1] * kz[i] + eps[i] * kz[i+1])

        if(polarization=='s'):
            for i in range(number - 1):
                r[i] = (kz[i] - kz[i+1]) / (kz[i] + kz[i+1])
            for i in range(number - 1):
                t[i] = 2 * kz[i] / (kz[i] + kz[i+1])            


        # Propagation matrix
        Pr = np.zeros((number-2, 2, 2), dtype=np.complex128)

        for i in range(Pr.shape[0]):
            phase = (1j * (2 * np.pi ) * kz[i+1] / wavelength) *  thickness[i]
        #    print(phase)
            Pr[i] = np.diag([np.exp(-phase), np.exp(phase)])


        # Refraction matrix
        Rfr = np.ndarray((number-1, 2, 2), dtype=np.complex128)

        for i in range(Rfr.shape[0]):
            Rfr[i] = np.array([[1, r[i]], [r[i], 1]], dtype=np.complex128)   
 

        S = Rfr[0]

        #print(f'kz{kz}\n r{r}\n t{t}\n Pr{Pr}\n Rfr{Rfr}\n')

        for i in range(number-2):
            S = S@(Pr[i]@Rfr[i+1])
            if np.any(np.isnan(S)):
                raise ValueError("Overflow in matrix exponent, probably because of wrong units of d.")
        #    print(f'S2{Pr[i]@Rfr[i+1]}')
        #    print(f'S{S}')

        # to calculate S-matrix you need to divide S by tij (in calculation of R tij vanishes)

        # reflection coefficient
        R = (np.abs(S[1, 0] / S[0, 0]))**2
        return R



print(abs(reflect_n(0, np.array([]), np.array([1, 10000000]), np.array([0, 0]), 22)))




