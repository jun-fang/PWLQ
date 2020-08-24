# This file is part of PWLQ repository.
# Copyright (c) Samsung Semiconductor, Inc.
# All rights reserved.

# specify your own imagenet data path
DATA='/home/jun/dataset/imagenet'

for MODEL in 'resnet50' 
do
    # evaluate the pretrained model with or w/o folding batch normalization
    python3 main.py -data=$DATA -a $MODEL
    python3 main.py -data=$DATA -a $MODEL -fbn

    # calibrate the activation ranges
    echo $'get activation stats for' $MODEL $'\n'
    python3 main.py -data=$DATA -a $MODEL -fbn -gs
    
    # compare PWLQ and uniform quantization  
    for WQ in 'uniform' 'pw-2' 
    do 
        for BIT in 8.0 6.0 4.0
        do 
            for BC in 'f' 't' 
            do
                for ACT in 'top_10'
                do 
                    if [[ $WQ -eq 'uniform' ]] 
                    then
                        python3 main.py -data=$DATA -a $MODEL -fbn -quant -ab=8.0 -aq=$ACT -wb=$BIT -wq='uniform' -bc=$BC
                    else
                        python3 main.py -data=$DATA -a $MODEL -fbn -quant -ab=8.0 -aq=$ACT -wb=$BIT -wq='pw-2' -bkp='norm' -appx='t' -bc=$BC
                    fi
                done
            done
        done
    done
done