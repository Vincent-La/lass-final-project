# LASS CMSC723 Final Project

+ Original [README](dcase_README.md)


## Setup One-Peace Submodule
```
cd ONE-PEACE && \
git submodule init && \
git submodule update
```



## LASS Environment Setup


Create envirnonment with conda/mamba
```
conda env create -f environment.yml && \
conda activate LASS
```

(Optional) If using mamba/micromamba you can run this to get Jupyter to detect the kernel for your LASS environment
```
python -m ipykernel install --user --name LASS
```

Pip install the rest of the ONE-PEACE dependencies
```
pip install -r requirements.txt
```

Then, try to run [`test.ipynb`](test.ipynb)


## Baseline Model Evaluation Setup


Environment Setup (**can skip this and use LASS environment**)
```
conda env create -f audiosep_environment.yml && \
conda activate AudioSep
```

Download Dcase Validation Dataset
```
wget https://zenodo.org/records/10886481/files/lass_validation.zip && \
unzip lass_validation.zip
```

Download baseline Audiosep model checkpoint and CLAP model checkpooint
```
mkdir checkpoint && \
cd checkpoint && \
wget https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt && \
wget https://zenodo.org/records/10887460/files/audiosep_16k,baseline,step%3D200000.ckpt && \
mv audiosep_16k\,baseline\,step\=200000.ckpt audiosep_baseline.ckpt
```


Run Eval script
```
# Using SLURM job
sbatch submit.sh

# or directly from python script
python dcase_evaluator.py
```

Output should match [audiosep_baseline](audiosep_baseline)