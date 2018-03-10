import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import json
from sys import exit
from sys import stdout
import time
import os
from shutil import copy2
from iext import Iext

##############################################
#### Set initial parameters
##############################################

f = open('params.json','r')
params = json.load(f)
f.close()

savefile = False

# Number of steps
M = params['nt']
# Timestep:
# I initially used 1 ns, but then found that Kunert used 1 µs.
dt = params['dt']
# Time 0
t0 = params['t0']
# Final time
t1 = t0 + M*dt

# Load the connectome and neurotransmitters, +1 excitatory, -1 inhibitory
f = open('aconnectome.json','r')
content = json.load(f)
Gsyn = np.array(content['chemical']).T
Ggap = np.array(content['electrical']).T
Neurotrans = np.array(content['chemical_sign'])
f.close()

# If non-interacting, set all elements to zero
if params['interacting'] == 0:
    Gsyn[:,:] = 0
    Ggap[:,:] = 0

# Number of neurons
N = len(Neurotrans)

# Cell
C = params['C'] # Membrane capacitance [F]
Gcell = params['Gcell'] # Leakage conductance of membrane [S]
Ecell = params['Ecell'] # Leakage potential [V]

# Chemical synapses
gsyn = params['gsyn'] # "conductivity" of chemical synapse [Siemens]
ar = params['ar'] # activation rate of synapses [s^-1]
ad = params['ad'] # deactivation rate of synapses [s^-1]
beta = params['beta'] # width of synaptic activation [V^-1]
esynexc = params['esynexc'] # reverse potential for excitatory synapses
esyninh = params['esyninh'] # reverse potential for inhibitory synapses

# Electrical synapses
ggap = params['ggap'] # conductivity of electrical synapse [Siemens]

# Build the Esyn array of the synaptic reverse potentials
# The index is presynaptic neuron, which determines the neurotransmitter and
# hence the sign of the synapse.
Esyn = 0.5*(Neurotrans+1)*esynexc - 0.5*(Neurotrans-1)*esyninh


##############################################
#### Define functions
##############################################

## Equilibrium

# Synaptic activations at rest, closed form
def Seq(ar=ar, ad=ad):
    return 0.5*ar/(0.5*ar+ad)
    
# Resting membrane potentials, to be calculated self-consistently
def Veq(V, S, Ggap=Ggap, Gsyn=Gsyn, Gcell=Gcell, ggap=ggap, gsyn=gsyn, \
        Esyn=Esyn, Ec=Ecell):
    VV = np.repeat([V], V.shape, 0)
    
    Y = Ec \
        - np.sum( Gsyn*gsyn*S/Gcell*(VV.T-Esyn[None,:]), axis=1 ) \
        - np.sum( Ggap*ggap/Gcell*(VV.T-V[None,:]), axis=1 ) 
    
    return Y


## Dynamics

# External current
# Use function Iext imported from iext module, so that the code can be stored
# with the plot of the results.

# System of differential equations
# Y' = f(t,Y), where Y is [Voltages,Synaptic activations]
def fY(t, Y, Vth, Iextbuff, Iext=Iext, Ggap=Ggap, Gsyn=Gsyn, Gcell=Gcell, \
        ggap=ggap, gsyn=gsyn, beta=beta, Esyn=Esyn, Ec=Ecell, ar=ar, ad=ad, \
        N=N):
    #The first N elements of Y are voltages, the last N are synaptic activations
    
    V = Y[:N]
    S = Y[N:]
    
    VV = np.repeat([V], V.shape, 0)
    Iext(t,Iextbuff)
    
    Vdot = 1./C * ( - Gcell*(V-Ec) \
            - np.sum( Ggap*ggap*(VV.T-V[None,:]), axis=1 ) \
            - np.sum( Gsyn*gsyn*S*(VV.T-Esyn[None,:]), axis=1 ) \
            +  Iextbuff)
    
    Sdot = ar / ( 1.0 + np.exp(-beta*(V-Vth)) ) * (1.-S) - ad*S
    
    Ydot = np.append(Vdot,Sdot)
    
    return Ydot
    

    
##############################################
#### Do the calculation
##############################################


#######################################
##### EQUILIBRIUM
#######################################

print("\n\n------- Calculating the resting properties.")

# Initial guess for the voltages
# Load them from previously saved file. If file does not exist, initialize
# at Ec.
try:
    f = open('V0.json','r')
    V = np.array(json.load(f))
    f.close()
except:
    V = np.ones(N)*Ecell

## The synaptic activations at rest have a closed form.
S = Seq()*np.ones(N,dtype=np.float64)

## Self-consistently determine the resting membrane potentials
i = 0
maxit = 100000
# Help convergence by taking the weighted average of the previous result and 
# the new one, with a very tiny weight for the latter. This damps the huge 
# oscillations that can build up around the "converged" average value.
while True:
    Vold = np.copy(V)
    damp = 1e-3
    V =  (Vold + damp*Veq(Vold, S))/(1.+damp)
    dV = np.sum(np.abs((V-Vold)/Vold))
    i+=1
    stdout.write("dV = "+str(dV)+"\t i = "+str(i)+"\r")
    if (dV<1e-15 or i>maxit): print("\n");break
    
if i<maxit: 
    print("The self-consistency condition converged in "+str(i)+" iterations.")
else:
    exit("The self-consistency did not converge.")
    
# Save it for next time
f = open('V0.json','w')
json.dump(V.tolist(),f)
f.close()
    
# Copy the resting voltages in V0.
V0 = np.copy(V)
    
# The half-activations of the synapses are set to be the resting potentials.
Vth = np.copy(V)

# Build the full set of degrees of freedom
Y0 = np.append(V0,S)

######################################
#### DYNAMICS
######################################

print("\n------- Starting with the dynamics.")

# Define the buffer for the external current. Cannot be pre-populated as
# (times, neurons) array because it would be huge.
Iextbuff = np.zeros(N)

r = ode(fY).set_integrator('vode',method='bdf',nsteps=1000)
r.set_initial_value(Y0,t0).set_f_params(Vth,Iextbuff)

evolvingY = []
T = []
Iextsave = []
zi = 0
undersample = 100
while r.successful() and r.t < t1:
	r.integrate(r.t+dt)
	if zi%undersample == 0:
		evolvingY.append(np.array(r.y))
		T.append(r.t)
		Iextsave.append(Iextbuff.tolist())
		stdout.flush()
		stdout.write("\r zi = "+str(zi))
	zi += 1
print("\n")

evolvingY = np.array(evolvingY)

######################################
#### SAVE OUTPUT
######################################

#______CREATE FOLDER WHERE TO SAVE THE DATA IF NEEDED_______#
folder = "Results/"+time.strftime("%Y-%m-%d")+"/"
filename = time.strftime("%H-%M-%S")
if not os.path.exists(folder):
	os.makedirs(folder)

# Save undersampled results
if savefile:
    f = open(folder+filename,'w')
    json.dump({'Y0': Y0.tolist(), 'Y': evolvingY.tolist(), 'T': T, \
             'Iext': Iextsave},f)
    f.close()
    
# Save parameters.
f = open(folder+filename+"-params.json",'w')
json.dump(params, f, sort_keys=True, indent=4, separators=(',', ': '))
f.close()

# Save code of the function Iext.
copy2('iext.py', folder+filename+"-iext.py")

# Plot evolution of variation of membrane voltages and synaptic activation.
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(T,evolvingY[:,:300]-Y0[:300])
ax1.set_xlabel("Time (s)",fontsize=14)
ax1.set_ylabel(r"$\Delta V\,\, (V)$", fontsize=18)

ax2.plot(T,evolvingY[:,300:])
ax2.set_xlabel("Time (s)",fontsize=14)
ax2.set_ylabel("Synaptic activation", fontsize=14)

plt.tight_layout()
plt.savefig(folder+filename+".png",bbox_inches='tight')
