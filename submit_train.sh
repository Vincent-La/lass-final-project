#!/bin/bash

#SBATCH --job-name=audiosep_onepeace_train                     # sets the job name
#SBATCH --output=audiosep_onepeace_train.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=audiosep_onepeace_train.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=24:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=class
#SBATCH --qos=high                                 # set QOS, this will determine what resources can be requested
#SBATCH --account=class
#SBATCH --gres=gpu:rtxa5000:4

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=4                                              
#SBATCH --ntasks-per-node=4                                      
#SBATCH --mem=128gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
micromamba activate LASS

srun python train.py --workspace . \
                     --config_yaml config/audiosep_onepeace.yaml \
                     --one_peace_checkpoint_path /fs/nexus-scratch/vla/finetune_al_retrieval.pt

wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
