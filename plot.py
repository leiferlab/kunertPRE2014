import numpy as np
import matplotlib.pyplot as plt
import json

f = open('output','r')
cont = json.load(f)
Y = np.array(cont['Y'])
T = cont['T']
Y0 = np.array(cont['Y0'])
f.close()

f = open('V0.json','r')
V = np.array(json.load(f))
f.close()

#plt.imshow((Y[:,:145]-Y[0,:145]),aspect='auto',cmap='gist_ncar',interpolation='none')#,vmin=-10e-3,vmax=10e-3)
plt.plot(T,Y[1:,:300]-Y[0,:300])#;plt.ylim(0.,2e-3)
#plt.plot(T,Y[1:,44]-Y[0,44])
plt.show()

