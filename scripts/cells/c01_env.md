#### Окружение

Локально Apple M4, 16 ГБ, без CUDA. Окружение через uv (Python 3.12): coqui-tts[codec]==0.28.0 (поддерживаемый форк idiap/coqui-ai-TTS), transformers==4.57.6 (с 5.x генерация XTTS ломается), demucs, faster-whisper, librosa, soundfile. ffmpeg из Homebrew нужен для torchcodec

Подготовка данных, генерация для проверок, WavLM и whisper работают на MPS. Дообучение в 16 ГБ общей памяти M4 не помещается (441M обучаемых параметров + AdamW, OOM уже на первом шаге), поэтому обучение и генерация трёх моделей сделаны на Kaggle (Tesla T4): kaggle/code/run.py, данные и скрипты загружены приватным датасетом

Trainer из coqui работает только с CUDA, поэтому цикл обучения свой (scripts/train_xtts.py). Модель, датасет, лосс и оптимизатор при этом берутся из GPTTrainer, как в рецепте recipes/ljspeech/xtts_v2/train_gpt_xtts.py

Код (скрипты, ячейки ноутбука, логи, метрики, генерации, обучающие наборы): [github.com/andreygetmanov/dls_homework_5](https://github.com/andreygetmanov/dls_homework_5). Полные аудиодорожки, из которых нарезаны данные (дорожки «Хоббита», 8 серий South Park в озвучке MTV), в репозиторий не влезают и лежат в датасете Kaggle [dls-hw5-raw-audio](https://www.kaggle.com/datasets/andreygetmanov/dls-hw5-raw-audio). Обучение и генерация на Kaggle: ноутбуки [dls-hw5-xtts-smaug](https://www.kaggle.com/code/andreygetmanov/dls-hw5-xtts-smaug) и [dls-hw5-xtts-cartman](https://www.kaggle.com/code/andreygetmanov/dls-hw5-xtts-cartman), данные к ним [dls-hw5-smaug](https://www.kaggle.com/datasets/andreygetmanov/dls-hw5-smaug) и [dls-hw5-cartman](https://www.kaggle.com/datasets/andreygetmanov/dls-hw5-cartman)
