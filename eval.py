from dcase_evaluator import DCASEEvaluatorAnalysis
from models.audiosep import AudioSep
from models.one_peace_encoder import ONE_PEACE_Encoder

import argparse
import os
from utils import parse_yaml, load_ss_model

def eval(evaluator,
         encoder_checkpoint_path = None, 
         ssnet_checkpoint_path = None, 
         config_yaml=None, 
         device = "cuda"):
    
    configs = parse_yaml(config_yaml)
    
    # ONE_PEACE modelhub expects some paths to be relative to this dir
    os.chdir('ONE-PEACE/')
    # TODO:path in shared scratch dir for now..., move to class project dir whenever we get that
    query_encoder = ONE_PEACE_Encoder(pretrained_path=encoder_checkpoint_path)
    os.chdir('..')

    # put ONE-PEACE model in eval model (probably unecessary)
    query_encoder.model.model.eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=ssnet_checkpoint_path,
        query_encoder=query_encoder
    ).to(device)


    print(f'-------  Start Evaluation  -------')

    # evaluation 
    SISDR, SDRi, SDR = evaluator(pl_model)
    msg_clotho = "SDR: {:.3f}, SDRi: {:.3f}, SISDR: {:.3f}".format(SDR, SDRi, SISDR)
    print(msg_clotho)

    print('-------------------------  Done  ---------------------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for AudioSep model",
    )

    parser.add_argument(
        "--encoder_checkpoint_path",
        type=str,
        required=True,
        help="Path of pretrained checkpoint for QueryEncoder (ONE-PEACE)",
    )

    parser.add_argument(
        '--ssnet_checkpoint_path',
        type=str,
        required=True,
        help = "Path of pretrained checkpoint for Seperation Network (ResUNet)"
    )

    args = parser.parse_args()

    dcase_evaluator = DCASEEvaluatorAnalysis(
        sampling_rate=16000,
        eval_indexes='lass_synthetic_validation.csv',
        audio_dir='lass_validation',
    )

    # checkpoint_path=
    eval(dcase_evaluator,
         encoder_checkpoint_path = args.encoder_checkpoint_path,
         ssnet_checkpoint_path = args.ssnet_checkpoint_path,
         config_yaml = args.config_yaml,
         device = "cuda",)
