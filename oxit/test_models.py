import os
import time
from TTS.api import TTS


def main():
    '''Test TTS quality of the different models.'''
    os.environ['TTS_HOME'] = 'oxit/assets/'

    ljspeech_models = [
        'tts_models/en/ljspeech/tacotron2-DDC',
        'tts_models/en/ljspeech/tacotron2-DDC_ph',
        'tts_models/en/ljspeech/glow-tts',  # fast and not bad
        'tts_models/en/ljspeech/speedy-speech',
        'tts_models/en/ljspeech/tacotron2-DCA',
        'tts_models/en/ljspeech/vits',
        'tts_models/en/ljspeech/vits--neon',
        'tts_models/en/ljspeech/fast_pitch',
        'tts_models/en/ljspeech/overflow',
        'tts_models/en/ljspeech/neural_hmm',
    ]

    ljspeech_vocoders = [
        'vocoder_models/en/ljspeech/multiband-melgan',
        'vocoder_models/en/ljspeech/hifigan_v2',
        'vocoder_models/en/ljspeech/univnet',
    ]

    text = 'Hello world hello world!'
    text2 = 'This is a sample text. I hope you like what you are hearing.'

    samples_dir = 'oxit/samples'
    if not os.path.exists(samples_dir):
        os.mkdir(samples_dir)

    times = []
    for model_name in ljspeech_models[:1]:
        tts = TTS(model_name)
        tts.tts(text)
        t1 = time.perf_counter()
        tts.tts(text)
        t2 = time.perf_counter()
        tts.tts(text2)
        t3 = time.perf_counter()
        m = model_name.split('/')[-1]
        tts.tts_to_file(text, file_path=os.path.join(samples_dir, f'{m}---1.wav'))
        tts.tts_to_file(text2, file_path=os.path.join(samples_dir, f'{m}---2.wav'))
        times.append((tts, t2 - t1, t3 - t2))

    for tts, t1, t2 in times:
        print(tts.model_name, t1, t2)


if __name__ == '__main__':
    main()
