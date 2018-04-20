# Definition of external current
# 1 pA [10 pA / (10 pS) in the paper, strange way they use to defining the scale] seems to be the value used by Kunert in his Fig 3 (2014), even though he does that strange normalization. It is set as the default value of the Imax argument. Use it as a reference for the amplitude of the current.

def Iext(t, Iextbuff, Imax=1e-12):
    ## Example
    if t>0.001 and t<0.004:
        Iextbuff[44] = Imax 
    else:
        Iextbuff[44] = 0.0
        
    ############################
    ## Other examples of stimuli
    ############################
    
    ## Other example: constant current
    #Iextbuff[146] = 3e-8 #PLM 145-146
    
    ## This stimulus should reproduce Figure 3 of the paper.
    #if t<0.25 or (t>0.5 and t<0.75):
    #    Iextbuff[44] = Imax #10 pA / (10 pS) 
    #else:
    #    Iextbuff[44] = 0.0
    
    return 1
