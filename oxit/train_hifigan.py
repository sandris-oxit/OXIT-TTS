import os
from multiprocessing import freeze_support
from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN


def main():
    root_path = 'oxit'
    output_path = os.path.join(root_path, 'runs')
    data_path = os.path.join(root_path, 'datasets', 'hungarian-single-speaker-tts', 'egri_csillagok')
    feature_path = os.path.join(root_path, 'datasets', 'mel')

    config = HifiganConfig(
        batch_size=32,
        eval_batch_size=32,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=True,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        lr_gen=1e-4,
        lr_disc=1e-4,
        data_path=data_path,
        feature_path=feature_path,
        output_path=output_path,
    )

    # init audio processor
    ap = AudioProcessor.init_from_config(config)

    # load training samples
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

    # init model
    model = GAN(config, ap)

    # init the trainer and 🚀
    trainer = Trainer(TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()


if __name__ == '__main__':
    freeze_support()
    main()
