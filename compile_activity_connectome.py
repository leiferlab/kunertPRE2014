import numpy as np, matplotlib.pyplot as plt
import os, sys, json
import pumpprobe as pp

merge = "--no-merge" not in sys.argv
inverted = "--inverted" in sys.argv
unc31 = "--unc31" in sys.argv
add_s_folder = "_unc31" if unc31 else ""

aconn_ds_i = None
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--aconn-ds-i": aconn_ds_i = int(sa[1])

if aconn_ds_i is None:
    if not inverted:
        folder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2"+add_s_folder+"/"
    else:
        folder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2_inverted"+add_s_folder+"/"
else:
    folder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2"+"_aconn-ds-"+str(aconn_ds_i)+add_s_folder+"/"
print("using",folder)
print("merging" if merge else "not merging")

funa = pp.Funatlas(merge_bilateral=False,merge_dorsoventral=False,
                   merge_numbered=False,merge_AWC=False)
                   
funa2 = pp.Funatlas(merge_bilateral=merge,merge_dorsoventral=False,
                   merge_numbered=False,merge_AWC=True)

act_conn = np.zeros((funa2.n_neurons,funa2.n_neurons))
corr = np.zeros((funa2.n_neurons,funa2.n_neurons))
count = np.zeros((funa2.n_neurons,funa2.n_neurons))
count_corr = np.zeros((funa2.n_neurons,funa2.n_neurons))

#ADDED FROM KUNERT_VS_CORRELATIONS_OPTIMIZE_CORRELATION (1ST STEP)
corr_individual = np.zeros((funa2.n_neurons,funa2.n_neurons,funa2.n_neurons))*np.nan
count_corr_individual = np.zeros((funa2.n_neurons,funa2.n_neurons,funa2.n_neurons))

corr_ = []

for neu_j in np.arange(funa.n_neurons):
    print(neu_j)
    neu_id = funa.neuron_ids[neu_j]
    
    #f = open(folder+neu_id,"r")
    #content = json.load(f)
    #f.close()
    #y = np.array(content["Y"])
    y = np.loadtxt(folder+neu_id+".txt")
    
    #Get the ai in the merged funatlas
    neu_j2 = funa2.ids_to_i([neu_id])
    
    y = y[:y.shape[0]//2]
    y -= y[:,0][:,None]
    
    y_stim = y[neu_j]
    r = np.corrcoef(y)
    
    for neu_i in np.arange(funa.n_neurons):
        yargmax = np.argmax(np.abs(y[neu_i]))
        y_resp = y[neu_i]
        
        #Get the ai in the merged funatlas
        neu_i2 = funa2.ids_to_i([funa.neuron_ids[neu_i]])
        act_conn[neu_i2,neu_j2] += y[neu_i,yargmax]
        count[neu_i2,neu_j2] += 1
        for neu_k in np.arange(funa.n_neurons):
            neu_k2 = funa2.ids_to_i([funa.neuron_ids[neu_k]])
            corr[neu_i2,neu_k2] += r[neu_i,neu_k]#np.corrcoef([y[neu_i],y[neu_k]])[0,1]
            count_corr[neu_i2,neu_k2] += 1
            if np.isnan(corr_individual[neu_j2,neu_i2,neu_k2]):
                corr_individual[neu_j2,neu_i2,neu_k2] = r[neu_i,neu_k]
                count_corr_individual[neu_j2,neu_i2,neu_k2] += 1
            else:
                corr_individual[neu_j2,neu_i2,neu_k2] += r[neu_i,neu_k]
                count_corr_individual[neu_j2,neu_i2,neu_k2] += 1

act_conn = act_conn/count
corr = corr/count_corr
corr_individual = corr_individual/count_corr_individual
add_s = "" if merge else "_no_merge"
np.savetxt(folder+"activity_connectome"+add_s+".txt",act_conn)
np.savetxt(folder+"activity_connectome_correlation"+add_s+".txt",corr)
np.save(folder+"activity_connectome_correlation_individual"+add_s+".npy",corr_individual)
act_conn = funa2.reduce_to_head(act_conn)

sorter_i = sorter_j = np.arange(act_conn.shape[-1])
lim = None

cmax = np.max(act_conn)
cmin = np.min(act_conn)
vmax = max(cmax,-cmin)
vmin = -vmax

    
fig1 = plt.figure(1,figsize=(12,10))
ax = fig1.add_subplot(111)
im = ax.imshow(act_conn,cmap='coolwarm',vmax=vmax,vmin=vmin,interpolation="nearest")
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))
ax.set_xticklabels(funa2.head_ids[sorter_j],fontsize=5,rotation=90)
ax.set_yticklabels(funa2.head_ids[sorter_i],fontsize=5)
ax.set_xlim(-0.5,lim)
fig1.savefig(folder+"AAA_intensity_map"+add_s+".png",dpi=300,bbox_inches="tight")
plt.show()
