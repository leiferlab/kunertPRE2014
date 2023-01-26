for ((i=0;i<4;i+=1))
do
    echo "running Kunert with aconn_ds_i $i"
    python kunert3.py 5e-10 --aconn-ds-i:$i
done
