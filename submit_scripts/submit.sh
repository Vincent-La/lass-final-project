#!/bin/bash

#SBATCH --job-name=audiosep_baseline                     # sets the job name
#SBATCH --output=audiosep_baseline.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=audiosep_baseline.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=02:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --partition=class
#SBATCH --qos=high                                 # set QOS, this will determine what resources can be requested
#SBATCH --account=class
#SBATCH --gres=gpu:rtxa5000:1

#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              
#SBATCH --ntasks-per-node=1                                      
#SBATCH --mem=64gb                                               # (cpu) memory required by job; if unit is not specified MB will be assumed

module load cuda

source ~/.bashrc
# eval "$(micromamba shell hook --shell bash)"
# micromamba activate AudioSep
micromamba activate LASS

python dcase_evaluator.py


wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
