# kunertPRE2014
Reproducing the calculations in Kunert et al., PRE 89 052805 (2014).

The main script is kunert.py. The first time you run it, it will save the equilibrium membrane potentials in the file V0.json, so that it does not have to calculate them every time.

## Input
To run a simulation you must provide the external stimuli to neurons I^ext_i coded as a function of time in the Iext function in the iext.py file. This file will be copied with the results, so that you have a record of the stimulus you applied.
The function Iext takes the time t as a first argument, a buffer/array Iextbuff in which the current at time t is written, and an optional argument Imax. You can use Imax as a reference value to use as amplitude of the current(s), since it seems to match what the authors used in the paper for figure 3. Iextbuff is an array and the i-th element corresponds to the external current you want to inject in neuron i at time t. Use the file neurons.txt to find the indexes of the neurons you want to stimulate.

The code also takes in a params.json file which contains some parameters. You can use the default ones.

## Output
The results of the simulations are saved in Results/yyyy-mm-dd with filenames that start with hh-mm-ss.
- hh-mm-ss is a json file containing Y0 (the equilibrium variables), Y (the time-evolving variables, undersampled in time otherwise the file becomes huge), and T (the undersampled time points). The file contains also a portion of Iext, but it is there just for debugging, so ignore it. The first 300 elements (or rows, for Y) of Y0 and Y are the membrane potentials, the last 300 elements are the synaptic activations (in this model, there is one synaptic activation for all the outgoing synapses of one specific neuron).
- hh-mm-ss.png contains a preview plot of the membrane potentials and of the synaptic activations.
- hh-mm-ss-iext.py is the copy of the Iext function that you used as stimulus.
- hh-mm-ss-params.json contains the parameters from the params.json file you used.

If there are no files saved, check that the savefile variable in the code is set to True.
