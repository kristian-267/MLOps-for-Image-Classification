 #!/bin/sh
 #BSUB -q gpua100
 #BSUB -gpu "num=2"
 #BSUB -J Train ResNeSt
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

 module load python3/3.7.10
 module load cuda/11.5
 module load cudnn
 module load ffmpeg

 pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

 echo "Running script..."
 make train
