import sunstone as sun
import numpy as np

class Stereo:
    """
    Class for geometric 3d-Stokes vector support. 
    The vector consist of 10 real numbers.
    The first 3 components are position coordinates P, the next 3 are orientation normalized componentes A. Final 4 are the components of Stokes vector. 
    
    Unphysical Stokes vector are allowed however their creation rises a type warning. Operations involving pure polarization states will use only the pure component of the current Stokes vector. Calling these methods on totally unpolarized light ( :math:`S = (1,0,0,0)`) will raise a type warning and nullify the output.

    :param d: Data for the 10 components of the Stereo vector.
    :type d: (10,) np.array, optional

    :return: Stereo vector.
    :rtype: Stereo
    """
    def __init__(self,d):
        self.P = d[0:3]
        self.A = d[3:6]
        self.A = self.A/np.sqrt(np.sum(self.A**2))

        self.S = sun.Stokes(d=d[6:])
    def forward(self):
        """Moves the vector a distance s, along the propagation direction.
        Negative s will move the vector against the propagation direction.

        :param s: Length of path traversed. 
        """
        self. P = self.P + s*self.A 