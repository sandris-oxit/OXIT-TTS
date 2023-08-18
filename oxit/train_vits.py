import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from multiprocessing import freeze_support
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def main():
    root_path = 'oxit'
    output_path = os.path.join(root_path, 'runs')
    dataset_path = os.path.join(root_path, 'datasets', 'hungarian-single-speaker-tts')
    phoneme_cache_path = os.path.join(root_path, 'datasets', 'phoneme_cache-hutts')
    dataset_config = BaseDatasetConfig(formatter='hungarian_tts', meta_file_train='transcript.txt', path=dataset_path)
    audio_config = VitsAudioConfig(sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None)

    config = VitsConfig(
        audio=audio_config,
        batch_size=32,
        eval_batch_size=32,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner='phoneme_cleaners',
        use_phonemes=True,
        phoneme_language='hu',
        phoneme_cache_path=phoneme_cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()


if __name__ == '__main__':
    freeze_support()
    main()
