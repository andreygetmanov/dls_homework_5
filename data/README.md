Полные аудиодорожки (data/raw: 8 серий South Park S01 в озвучке MTV, дорожки «Хоббита» со Смаугом, ролики с Мерил Стрип) в репозиторий не входят: файлы больше лимита GitHub и защищены авторским правом. Они лежат в приватном датасете Kaggle andreygetmanov/dls-hw5-raw-audio, доступ по ссылке: https://www.kaggle.com/datasets/9e1089c3797bb23110060d0ff48a56d5d68c950b9dad9249fc3f9bb97a719618

    kaggle datasets download andreygetmanov/dls-hw5-raw-audio -p data/raw --unzip

Вокал (data/separated) получается из них заново через demucs (htdemucs), промежуточные нарезки (data/segments/*/wavs) скриптами scripts/cartman_lines.py и scripts/smaug_from_srt.py
