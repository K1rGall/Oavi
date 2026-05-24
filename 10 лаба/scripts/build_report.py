#!/usr/bin/env python3
"""Build markdown report for lab 10 from generated artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build markdown report.")
    parser.add_argument("--results-json", type=Path, default=Path("results/recognition.json"))
    parser.add_argument("--output", type=Path, default=Path("report_lab10.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.results_json.read_text(encoding="utf-8"))

    predicted = data.get("predicted_sequence") or "-"
    expected = data.get("expected_sequence") or "-"
    errors = data.get("errors")
    confidence = data.get("confidence")
    segments = data.get("num_segments", 0)
    sample_rate = data.get("sample_rate", "-")

    errors_line = str(errors) if errors is not None else "не вычислялось (не задан эталон)"
    confidence_line = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "не вычислялась"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"""# Лабораторная работа №10: Обработка голоса (вариант 3)

**Дисциплина:** ОАВИ  
**Тема:** Анализатор речи  
**Дата формирования отчёта:** {now}

## 1. Цель работы
Реализовать пайплайн распознавания словарных единиц (0..9 и «плюс») в записи телефонного номера:
- построить спектрограмму записи на основе STFT с окном Ханна;
- сегментировать дорожку на отдельные слова;
- сопоставить сегменты с эталонами;
- получить цепочку символов, число ошибок и оценку достоверности.

## 2. Использованное ПО
- `ffmpeg` (генерация/подготовка аудио);
- `Python 3.11`;
- библиотеки: `numpy`, `scipy`, `matplotlib`.

## 3. Выполнение работы
1. Подготовлен алфавит из 11 эталонов (`0..9` и `+`) и запись телефонного номера.
   Для автоматического прогона в этом отчёте использован синтезированный голос (`ffmpeg flite`);
   для сдачи с собственным голосом используется `scripts/record_from_microphone.ps1`.
2. Для телефонной дорожки рассчитана спектрограмма STFT (окно Ханна, логарифмическая шкала частот).
3. Реализована процедура сегментации по кратковременной энергии.
4. Реализовано сопоставление сегментов с эталонами: MFCC + DTW.
5. Получен распознанный номер, рассчитаны ошибки и достоверность.

## 4. Результаты
- Частота дискретизации: **{sample_rate} Гц**
- Количество найденных сегментов: **{segments}**
- Ожидаемая последовательность: **{expected}**
- Распознанная последовательность: **{predicted}**
- Число ошибок (редакционное расстояние): **{errors_line}**
- Оценка достоверности: **{confidence_line}**

## 5. Скриншоты проделанной работы
### 5.1 Подготовка данных
![Список файлов датасета](screenshots/01_dataset_files.png)

### 5.2 Спектрограмма записи телефонного номера
![Спектрограмма](screenshots/02_spectrogram.png)

### 5.3 Сегментация телефонной дорожки
![Сегментация](screenshots/03_segmentation.png)

### 5.4 Матрица расстояний DTW
![DTW heatmap](screenshots/04_distance_heatmap.png)

### 5.5 Итоги распознавания (консоль)
![Итоги распознавания](screenshots/05_recognition_console.png)

## 6. Вывод
Построен и протестирован полный анализатор речи для варианта 3: спектральный анализ, сегментация и распознавание по словарю. Получены количественные метрики качества (ошибки и достоверность), результаты визуализированы и сохранены.
"""

    args.output.write_text(content, encoding="utf-8")
    print(f"Report created: {args.output}")


if __name__ == "__main__":
    main()
