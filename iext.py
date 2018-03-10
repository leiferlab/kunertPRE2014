# External current
# 1 pA seems to be the value used by Kunert in his Fig 3 (2014), even though he
# does that normalization.
def Iext(t, Iextbuff):
    if t<0.25 or (t>0.5 and t<0.75):
       Iextbuff[44] = 1e-12 #10 pA / (10 pS) 
    else:
        Iextbuff[44] = 0.0
    #Iextbuff[145] = 3e-8 #PLM
    
    return 1
