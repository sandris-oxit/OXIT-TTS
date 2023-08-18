import os
from multiprocessing import freeze_support
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def main():
    root_path = 'oxit'
    output_path = os.path.join(root_path, 'runs')
    dataset_path = os.path.join(root_path, 'datasets', 'hungarian-single-speaker-tts')
    phoneme_cache_path = os.path.join(root_path, 'datasets', 'phoneme_cache-hutts')
    dataset_config = BaseDatasetConfig(formatter='hungarian_tts', meta_file_train='transcript.txt', path=dataset_path)

    # INITIALIZE THE TRAINING CONFIGURATION
    # Configure the model. Every config class inherits the BaseTTSConfig.
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=32,  # increase this!
        num_loader_workers=4,  # persistent workers?
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner='phoneme_cleaners',  # check this?
        use_phonemes=True,
        phoneme_language='hu',
        phoneme_cache_path=phoneme_cache_path,
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )
    # check max_text_len?
    
    #config.log_model_step
    #config.save_step
    #config.plot_step  # fix these, log train stuff too!

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are passed to the config
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # INITIALIZE THE MODEL
    # Models take a config object and a speaker manager as input
    # Config defines the details of the model like the number of layers, the size of the embedding, etc.
    # Speaker manager is used by multi-speaker models.
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()


#os.environ['CUDA_VISIBLE_DEVICES']='1'
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
if __name__ == '__main__':
    freeze_support()
    main()
