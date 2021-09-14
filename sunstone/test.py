import sunstone as sun
import numpy as np

s =  sun.Stokes(10., 2., 3., 4.)
#print(s.dp())
#print(s.poincare())
testd = np.array(
    [ 
        [4,2,1,5],
        [2,1,2,1],
        [6,4,7,8],
        [7,8,9,4]
    ]
)
m = sun.Muller(testd).physical()


#print(m(s).data)
#print(sun.plot_poincare([sun.Jones(1.25,0.5+1j).stokes(),sun.Jones(1.25,0+1j).stokes()]))
#print(sun.Jones(1.25,0.5+1j).delta()/(6.28))