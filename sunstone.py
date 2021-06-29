import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy as sci
import scipy.linalg as lin

class Jones:
    """
    Class for Jones vector support. The entries of the vector are complex numbers in general. When casting to Stokes vectors type warnings are suppresed. The notation convention follows as Goldstein's *Polarized Light*.
    """
    def __init__(self,x=0,y=0,d=None):
        if d==None:
            self.x = x
            self.y = y
            self.data = np.array([x,y])
        else:
            self.x = d[0]
            self.y = d[1]
            self.data = np.array(d)
    def delta(self):
        """Calculates relative phase diffrence between vertical and horizontal components.

        :return: Delta angle.
        :rtype: float
        """
        aux = np.angle(self.data)
        delta = aux[1]-aux[0]
        return delta
    def wolf(self):
        """Calculates the Wolf coherency matrix.

        :return: Wolf matrix.
        :rtype: (2,2) array
        """
        return np.outer(self.data,np.conj(self.data))
    def stokes(self):
        """Converts the Jones vector to a Stokes vector.

        :return: Corresponding Stokes vector.
        :rtype: Stokes
        """
        x = self.x
        y = self.y
        xc = np.conj(x)
        yc = np.conj(y)
        with warnings.catch_warnings():
            # Casting complex numbers to float raises a warning
            # we supress these
            warnings.simplefilter("ignore")
            s0 = float(x*xc + y*yc)
            s1 = float(x*xc - y*yc)
            s2 = float(x*yc+y*xc)
            s3 = float((0+1j)*(x*yc-y*xc))
        return Stokes(s0,s1,s2,s3)

class Stokes:
    """
    Class for Stokes vector support. The vector consist of 4 real numbers. Unphysical Stokes vector are allowed however their creation rises a type warning. Operations involving pure polarization states will use only the pure component of the current Stokes vector. Calling these methods on totally unpolarized light ( :math:`S = (1,0,0,0)`) will raise a type warning and nullify the output.

    Builds a Stokes vector given each of its components `s0`,`s1`,`s2`,`s3` or an (,4) array `d`. 

    :param s0: First component of the stokes vector, defaults to 0
    :type s0: float, optional
    :param s1: Second component of the stokes vector, defaults to 0
    :type s1: float, optional
    :param s2: Third component of the stokes vector, defaults to 0
    :type s2: float, optional
    :param s3: Fourth component of the stokes vector, defaults to 0
    :type s3: float, optional
    :param d: Array with the components of the Stokes vector, defaults to None
    :type d: array, optional

    :return: Stokes vector.
    :rtype: Stokes
    """
    def __init__(self,s0 = 0,s1 = 0,s2= 0,s3 =0,d = None):
        if d == None:
            self.s0 = s0
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3

            self.d = np.array([s0,s1,s2,s3])
        else:
            self.s0 = d[0]
            self.s1 = d[1]
            self.s2 = d[2]
            self.s3 = d[3]
            self.d = d
        if self.s0 <= 0:
            warnings.warn("Non-physical Stokes vector (Null or negative intensity).")
        self.z = np.array([s1,s2,s3])
        self.znorm = np.sqrt(s1**2+s2**2+s3**2)
        # Minkowsky norm
        self.mnorm = s0**2-self.znorm**2
        if self.mnorm <0:
            warnings.warn("Non-physical Stokes vector (Outside Light Cone).")
    def is_pure(self):
        return self.mnorm == 0
    def pure(self):
        """Gets the pure polarization component of the vector.
        Raises a warning if the vector is totally unpolarized.

        :return: Pure polarization state Stokes vector.
        :rtype: Stokes
        """
        if self.znorm == 0:
                warnings.warn("Cannot purify totally unpolarized Stokes vector.")
                return self

        if not self.is_pure():
            return Stokes(self.znorm,self.s1,self.s2,self.s3)
        else:
            return self
    def dp(self):
        """Calculates the degree of polarization.

        :return: Degree of polarization.
        :rtype: float
        """
        return self.znorm / self.s0
    def chi(self):
        """
        Calculates the orientation angle of the polarization ellipse.

        :return: The orientation angle in radians.
        :rtype: float
        """
        if not self.is_pure():
            return self.pure().chi()
        else:
            return np.arcsin(self.s3/self.s0)/2.
    def psi(self):
        """
        Calculates the ellipticity angle of the polarization angle of the polarization ellipse.

        Returns:
            float: The ellipticity angle.
        """
        if not self.is_pure():
            return self.pure().psi()
        else:
            return np.arctan2(self.s2,self.s1)/2.
    def delta(self):
        """Relative phase difference of the pure component.

        :return: Phase difference.
        :rtype: Float
        """
        if not self.is_pure():
            return self.pure().delta()
        else:
            return np.arctan2(self.s3,self.s2)
    def poincare(self,unit = False):
        """
        Gets the coordinates of the Poincaré sphere represemtation of the Sokes vector.

        :param unit: Wether or not normalize the output vector. Defaults to true.
        :type unit: Boolean,optional.
        :return: [description]
        :rtype: [type]
        """
        n = 1
        if not unit:
            n=self.s0
        return n*np.array([np.cos(2*self.chi())*np.cos(2*self.psi()),
            np.cos(2*self.chi())*np.sin(2*self.chi()),
            np.sin(2*self.chi())])
    def plot_poincare(self):
        """Makes a 3d plot of the Stokes vector representation in the Poincaré sphere.
        """
        vec = self.poincare()
        # We draw the sphere on 3d axes
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Arrays of the parametrized sphere
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)

        # Meshes of the 3d coordinates of the wireframe
        x = np.outer(np.sin(u), np.sin(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.cos(u), np.ones_like(v))

        ax.plot_wireframe(x, y, z,linewidth=0.1)
        ax.quiver([0],[0],[0],[vec[0]],[vec[1]],[vec[2]])
        ax.set_xlabel("$S_1$")
        ax.set_ylabel("$S_2$")
        ax.set_zlabel("$S_3$")
        plt.show()

    def __add__(self, a):
        return Stokes(d = a.d + self.d)
    def __sub__(self, a):
        return Stokes(d = a.d - self.d)
    def __mul__(self, a):
        return Stokes(d = a*self.d)

class Muller:
    """
    The base class for Müller matrices. The underlying 4x4 matrix is kept under the attribute ``data``. 
    """
    def __init__(self,data):
        self.data = np.array(data)
        if(data.shape != (4,4)):
            warnings.warn("Muller matrix doesn't have the correct dimensions.")
    def eig(self):
        return lin.eig(self.data)
    def physical(self):
        """Implements physicality criterion for the Müller matrix. [TODO]
        """
        e = self.eig()
        # We get the maximum eigenvalue and the
        # associated eigenvector
        am = np.argmax(e[0])
        maxev = e[0][am]
        maxevec = e[1][am]
    def __call__(self, s: Stokes)-> Stokes:
        a = self.data @ s.data
        return Stokes(d = a)
    def __matmul__(self, s):
        a = self.data @ s
        return None

Gmat = np.eye(4)
Gmat[0][0] = -1
G = Muller(Gmat)

def plot_poincare(a):
    # We draw the sphere on 3d axes
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Arrays of the parametrized sphere
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    # Meshes of the 3d coordinates of the wireframe
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    ax.set_xlabel("$S_1$")
    ax.set_ylabel("$S_2$")
    ax.set_zlabel("$S_3$")
    ax.plot_wireframe(x, y, z,linewidth=0.1)
    try:
        # if we are given a list argument we plot each vector
        for s in a:
            vec = s.poincare()
            ax.quiver([0],[0],[0],[vec[0]],[vec[1]],[vec[2]])
    except TypeError:
        # otherwise plot the single vector
        vec = a.poincare()
        ax.quiver([0],[0],[0],[vec[0]],[vec[1]],[vec[2]])
    plt.show()
