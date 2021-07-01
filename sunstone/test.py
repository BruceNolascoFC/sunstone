import sunstone as sun
import numpy as np

s =  sun.Stokes(10., 2., 3., 4.)
#print(s.dp())
#print(s.poincare())

m = sun.Muller(np.eye((4)))
print(np.argmax(m.eig()[0]))
#print(m(s).data)
#print(sun.plot_poincare([sun.Jones(1.25,0.5+1j).stokes(),sun.Jones(1.25,0+1j).stokes()]))
#print(sun.Jones(1.25,0.5+1j).delta()/(6.28))