import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import json
from sys import exit
from sys import stdout

##############################################
#### Set initial parameters
##############################################

# Numer of neurons
N = 300
# Number of steps
M = 3
# Timestep
dt = 1
# Time 0
t0 = 0
# Final time
t1 = t0 + M*dt

# Cell
C = 1e-12 # Membrane capacitance [F]
Gcell = 1e-11*np.ones(N) # Leakage conductance of membrane [S]
Ecell = -35e-3*np.ones(N) # Leakage potential [V]

# Chemical synapses
gsyn = 1e-10 # "conductivity" of chemical synapse [Siemens]
ar = 1. # activation rate of synapses [s^-1]
ad = 5. # deactivation rate of synapses [s^-1]
beta = 125. # width of synaptic activation [V^-1]
esynexc = 0. # reverse potential for excitatory synapses
esyninh = -45e-3 # reverse potential for inhibitory synapses

# Electrical synapses
ggap = 1e-10

# Load the connectome
# and neurotransmitters, +1 excitatory, -1 inhibitory
f = open('../aconnectome/aconnectome.json','r')
content = json.load(f)
Gsyn = np.array(content['chemical'])
Ggap = np.array(content['electrical'])
Neurotrans = np.array(content['chemical_sign'])
f.close()

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
    Y = Ec 
    Y -= np.sum( Ggap/Gcell*ggap*(VV-V), axis=0 ) 
    Y -= np.sum( Gsyn/Gcell*gsyn*S*(VV-Esyn), axis=0 )
        
    return Y


## Dynamics

# System of differential equations
# Y' = f(t,Y), where Y is [Voltages,Synaptic activations]
def fY(t, Y, Vth, Iext, Ggap=Ggap, Gsyn=Gsyn, C=C, Gcell=Gcell, ggap=ggap, \
        gsyn=gsyn, beta=beta, Esyn=Esyn, Ec=Ecell, ar=ar, ad=ad):
    #The first N elements of Y are voltages, the last N are synaptic activations
    
    V = Y[:N]
    S = Y[N:]
    
    VV = np.repeat([V], V.shape, 0)
    Igap = 1./C * np.sum( Ggap*ggap*(VV-V), axis=0 )
    Isyn = 1./C * np.sum( Gsyn*gsyn*S*(VV-Esyn), axis=0 )
    Icel = 1./C * Gcell*(V-Ec)
    
    Vdot = -Icel-Igap-Isyn+Iext[int(t)]
        
    Sdot = ar / ( 1.0+np.exp(-beta*(V-Vth)) ) * (1.-S) - ad*S
    
    Ydot = np.append(Vdot,Sdot)
    
    return Ydot
    
    



##############################################
#### Do the calculation
##############################################

# Initial guess for the voltages
#V = -12e-3*np.ones(N)
f = open('V0.json','r')
V = np.array(json.load(f))
f.close()

#######################################
##### EQUILIBRIUM
#######################################

print("\n\n------- Calculating the resting properties.")

## The synaptic activations at rest have a closed form.
S = Seq()*np.ones(N)

## Self-consistently determine the resting membrane potentials
i = 0
maxit = 40000
# Help convergence by taking the average with the K previous steps. This damps
# the huge oscillations that can build up around the "converged" average value.
while True:
    Vold = V
    V =  Vold+ 1e-3*(Veq(Vold, S) - Vold) 
    dV = np.abs(V-Vold)
    i+=1
    if (all(dV<1e-6) or i>maxit): break
    
# Copy the resting voltages in V0.
V0 = np.copy(V)
if i<maxit: 
    print("The self-consistency condition converged in "+str(i)+" iterations.")
else:
    exit("The self-consistency did not converge.")

plt.plot(V0)
plt.savefig('V0.png')

f = open('V0.json','w')
json.dump(V0.tolist(),f)
f.close()

# The half-activations of the synapses are set to be the resting potentials.
Vth = np.copy(V)

# Build the full set of degrees of freedom
Y0 = np.append(V0,S)


######################################
#### DYNAMICS
######################################

print("\n------- Starting with the dynamics.")

Iext = np.zeros((M+1,N))
#Iext[1:,44] = 1e7 #44 is ASHR


r = ode(fY).set_integrator('dopri5',nsteps=1000)
r.set_initial_value(Y0,t0).set_f_params(Vth,Iext)

evolvingY = []
evolvingY.append(Y0)
T = []
zi = 0
undersample = 1
while r.successful() and r.t < t1:
	r.integrate(r.t+dt)
	if zi%undersample == 0:
		evolvingY.append(r.y)
		T.append(r.t)
		stdout.flush()
		stdout.write("\rCurrent time %d/%d" % (r.t,t1))
	zi += 1
print("\n")

f = open('evolvingY','w')
json.dump(np.array(evolvingY).tolist(),f)
f.close()
    
