# OXIT-TTS

Hungarian text-to-speech using [coqui-ai/TTS](https://github.com/coqui-ai/TTS).

## Installation

1. Download [Hungarian Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/hungarian-single-speaker-speech-dataset)
2. Install prerequisites: `eSpeak` or `eSpeak NG`, and add to `PATH`
3. Clone repository
4. Change branch to `hun`
5. Create virtual environment
6. Install `PyTorch` + `CUDA`
7. Install TTS: `python -m pip install -e .`

Test installation success: `python oxit/test_models.py`

## Training

Before starting the training process, check the dataset using the notebooks, and try to find the best audio parameters.

1. Train spectrogram model
```bash
python oxit\train_glowtts.py
```

2. Create spectrogram dataset
```bash
python TTS\bin\extract_tts_spectrograms.py --config_path run\config.json --checkpoint_path run\best_model.pth --output_path datasets\mel
```

3. Train vocoder model
```bash
python oxit\train_hifigan.py
```

4. Synthesize audio
```bash
tts --model_path run\best_model.pth --config_path run\config.json --vocoder_path run\best_model.pth --vocoder_config_path run\config.json --text "Ez egy próba mondat."
```

## Progress

#### Glow-TTS + HiFiGAN:

[Sample 1](samples/glowtts-hifigan1.wav)  
[Sample 2](samples/glowtts-hifigan2.wav)  
[Sample 3](samples/glowtts-hifigan3.wav)  

#### VITS:

[Sample 1](samples/vits1.wav)  
[Sample 2](samples/vits2.wav)  
[Sample 3](samples/vits3.wav)  

#### Tacotron2 + HiFiGAN/MelGAN:

...
