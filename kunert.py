import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import json
from sys import exit
from sys import stdout

##############################################
#### Set initial parameters
##############################################

# Number of steps
M = 1e6
# Timestep:
# I initially used 1 ns, but then found that Kunert used 1 µs.
dt = 1e-6 
# Time 0
t0 = 0
# Final time
t1 = t0 + M*dt

# Load the connectome and neurotransmitters, +1 excitatory, -1 inhibitory
f = open('aconnectome.json','r')
content = json.load(f)
Gsyn = np.array(content['chemical']).T
Ggap = np.array(content['electrical']).T
Neurotrans = np.array(content['chemical_sign'])
f.close()

# Number of neurons
N = len(Neurotrans)

# Cell
C = 1e-12 # Membrane capacitance [F]
Gcell = 1e-11 # Leakage conductance of membrane [S]
Ecell = -35e-3 # Leakage potential [V]

# Chemical synapses
gsyn = 1e-10 # "conductivity" of chemical synapse [Siemens]
ar = 1. # activation rate of synapses [s^-1]
ad = 5. # deactivation rate of synapses [s^-1]
beta = 125. # width of synaptic activation [V^-1]
esynexc = 0. # reverse potential for excitatory synapses
esyninh = -45e-3 # reverse potential for inhibitory synapses

# Electrical synapses
ggap = 1e-10 # conductivity of electrical synapse [Siemens]

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
# 1 pA seems to be the value used by Kunert in his Fig 3 (2014), even though he
# does that normalization.
def Iext(t, Iextbuff):
    #if t<0.25 or (t>0.5 and t<0.75):
    #   Iextbuff[44] = 1e-12 #10 pA / (10 pS) 
    #else:
    #    Iextbuff[44] = 0.0
    Iextbuff[145] = 3e-8 #PLM
    
    return 1
    

# System of differential equations
# Y' = f(t,Y), where Y is [Voltages,Synaptic activations]
def fY(t, Y, Vth, Iextbuff, Iext=Iext, Ggap=Ggap, Gsyn=Gsyn, Gcell=Gcell, ggap=ggap, \
        gsyn=gsyn, beta=beta, Esyn=Esyn, Ec=Ecell, ar=ar, ad=ad, N=N):
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
# Load them from previously saved file.
f = open('V0.json','r')
V = np.array(json.load(f))
f.close()

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

plt.plot(V0)
plt.savefig('V0.png')
plt.clf()
    
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
evolvingY.append(Y0)
T = []
zi = 0
undersample = 100
while r.successful() and r.t < t1:
	r.integrate(r.t+dt)
	if zi%undersample == 0:
		evolvingY.append(np.array(r.y))
		T.append(r.t)
		stdout.flush()
		stdout.write("\r zi = "+str(zi))
	zi += 1
print("\n")

f = open('evolvingY','w')
json.dump({'Y': np.array(evolvingY).tolist(), 'T': T},f)
f.close()
