import sunstone as sun
import numpy as np
pi = np.pi
#print(s.dp())
#print(s.poincare())
m = sun.Polarizer(0).d
print(sun.Analyzer(2.).d)

print(sun.Polarizer(pi-1).physical())
#print(m(s).data)
#print(sun.plot_poincare([sun.Jones(1.25,0.5+1j).stokes(),sun.Jones(1.25,0+1j).stokes()]))
#print(sun.Jones(1.25,0.5+1j).delta()/(6.28))
