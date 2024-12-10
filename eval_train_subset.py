from training_subset_analysis import TrainingSubsetAnalysis
from models.audiosep import AudioSep
import argparse
import os
from utils import parse_yaml, load_ss_model
from scipy.signal import spectrogram


def eval(evaluator,
         encoder_checkpoint_path = None, 
         ssnet_checkpoint_path = None, 
         config_yaml=None, 
         device = "cuda",
         encoder_type = None):

    
    assert encoder_type is not None, 'define encoder type'
    
    configs = parse_yaml(config_yaml)
    
    if encoder_type == 'ONE-PEACE':

        from models.one_peace_encoder import ONE_PEACE_Encoder
        # ONE_PEACE modelhub expects some paths to be relative to this dir
        os.chdir('ONE-PEACE/')
        # TODO:path in shared scratch dir for now..., move to class project dir whenever we get that
        query_encoder = ONE_PEACE_Encoder(pretrained_path=encoder_checkpoint_path)
        os.chdir('..')

        # put ONE-PEACE model in eval model (probably unecessary)
        query_encoder.model.model.eval()

    elif encoder_type == 'CLAP':
        from models.clap_encoder import CLAP_Encoder
        query_encoder = CLAP_Encoder(pretrained_path=encoder_checkpoint_path).eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=ssnet_checkpoint_path,
        query_encoder=query_encoder
    ).to(device)


    print(f'-------  Start Evaluation  -------')
    df_results = evaluator(pl_model)
    df_results.to_csv(f'{encoder_type}_training_subset.csv', index = None)
    print('-------------------------  Done  ---------------------------')
    # evaluation 
    

    
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
        help="Path of pretrained checkpoint for QueryEncoder (ONE-PEACE/CLAP)",
    )

    parser.add_argument(
        '--ssnet_checkpoint_path',
        type=str,
        required=True,
        help = "Path of pretrained checkpoint for Seperation Network (ResUNet)"
    )

    parser.add_argument(
        '--encoder_type',
        type=str,
        required=True,
        help= 'type of Query Encoder'
    )


    args = parser.parse_args()
    print(args)

   # Run evaluation on training subset + pull out per-sample metrics and similarity scores
    dcase_evaluator = TrainingSubsetAnalysis(
        sampling_rate=16000,
        eval_indexes='lass_training_subset.csv',
        audio_dir= '',        # use absolute paths in eval_indexes csv file
        output_dir = None,    # set to none to avoid making audio .wav files
        encoder_type=args.encoder_type,
        config_yaml = args.config_yaml
    )

    eval(dcase_evaluator,
         encoder_checkpoint_path = args.encoder_checkpoint_path,
         ssnet_checkpoint_path = args.ssnet_checkpoint_path,
         config_yaml = args.config_yaml,
         device = "cuda",
         encoder_type=args.encoder_type)
