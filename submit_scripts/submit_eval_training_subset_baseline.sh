#!/bin/bash

#SBATCH --job-name=audiosep_onepeace_eval                     # sets the job name
#SBATCH --output=audiosep_onepeace_eval.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=audiosep_onepeace_eval.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=10:00:00                                      # how long you would like your job to run; format=hh:mm:ss


#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger                                # set QOS, this will determine what resources can be requested
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1


#IGNORE SBATCH --partition=class
#IGNORE SBATCH --qos=high                                # set QOS, this will determine what resources can be requested
#IGNORE SBATCH --account=class
#IGNORE SBATCH --gres=gpu:rtxa5000:1


#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                             
#SBATCH --ntasks-per-node=1                                     
#SBATCH --mem=64gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda
source ~/.bashrc
micromamba activate LASS

srun python eval_train_subset.py --config_yaml config/audiosep_base.yaml \
                                 --encoder_checkpoint_path ./checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt \
                                 --ssnet_checkpoint_path checkpoint/audiosep_baseline.ckpt \
                                 --encoder_type CLAP


wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
