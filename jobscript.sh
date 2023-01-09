 #!/bin/sh
 #BSUB -q gpua100
 #BSUB -gpu "num=2"
 #BSUB -J Train ResNeSt
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 <loading of modules, dependencies etc.>
 echo "Running script..."
 python3 src/models/train_model.py
