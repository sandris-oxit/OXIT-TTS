import os
from multiprocessing import freeze_support
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def main():
    root_path = 'oxit'
    output_path = os.path.join(root_path, 'runs')
    dataset_path = os.path.join(root_path, 'datasets', 'hungarian-single-speaker-tts')
    phoneme_cache_path = os.path.join(root_path, 'datasets', 'phoneme_cache-hutts')

    dataset_config = BaseDatasetConfig(formatter='hungarian_tts', meta_file_train='transcript.txt', path=dataset_path)
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func='np.log',
        ref_level_db=20,
        preemphasis=0.0,
    )

    config = Tacotron2Config(
        audio=audio_config,
        batch_size=64,
        eval_batch_size=32,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        ga_alpha=0.0,
        decoder_loss_alpha=0.25,
        postnet_loss_alpha=0.25,
        postnet_diff_spec_alpha=0,
        decoder_diff_spec_alpha=0,
        decoder_ssim_alpha=0,
        postnet_ssim_alpha=0,
        r=2,
        attention_type='dynamic_convolution',
        double_decoder_consistency=False,
        epochs=1000,
        text_cleaner='phoneme_cleaners',
        use_phonemes=True,
        phoneme_language='hu',
        phoneme_cache_path=phoneme_cache_path,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = Tacotron2(config, ap, tokenizer)
    trainer = Trainer(TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()


if __name__ == '__main__':
    freeze_support()
    main()
