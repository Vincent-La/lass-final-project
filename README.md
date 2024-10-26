# LASS CMSC723 Final Project

+ Original [README](dcase_README.md)

## Baseline Model Evaluation Setup


Environment Setup
```
conda env create -f environment.yml && \
conda activate AudioSep
```

Download baseline Audiosep model checkpoint and CLAP model checkpooint
```
mkdir checkpoint && \
cd checkpoint && \
wget https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt && \
wget https://zenodo.org/records/10887460/files/audiosep_16k,baseline,step%3D200000.ckpt && \
mv audiosep_16k,baseline,step%3D200000.ckpt audiosep_baseline.ckpt
```

Download Dcase Validation Dataset
```
wget https://zenodo.org/records/10886481/files/lass_synthetic_validation.csv && \
wget https://zenodo.org/records/10886481/files/lass_validation.json && \
wget https://zenodo.org/records/10886481/files/lass_validation.zip && \
unzip lass_validation.zip
```

Run Eval script
```
# Using SLURM job
sbatch submit.sh

# or directly from python script
python dcase_evaluator.py
```

Output should match [audiosep_baseline](audiosep_baseline)