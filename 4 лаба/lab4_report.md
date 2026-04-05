# Отчёт по лабораторной работе №4
## Выделение контуров оператором Собеля (вариант 3)

**Исходные изображения:** `./input/*.png`  
**Результаты обработки:** `./output/`  
**Скрипт:** `./variant3_sobel.py`

---

## Цель работы
Реализовать выделение контуров на изображениях методом Собеля, получить карты градиентов `Gx`, `Gy`, модуль градиента `G` и бинарную карту границ.

---

## Используемый метод

### 1. Перевод в полутон
Если вход цветной, используется формула:

`Y = 0.299R + 0.587G + 0.114B`

### 2. Оператор Собеля 3x3
Маски:

`Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`

`Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]`

После свёртки:

`Gx = Gray * Kx`  
`Gy = Gray * Ky`

### 3. Модуль градиента и бинаризация
Модуль градиента вычисляется как:

`G = sqrt(Gx^2 + Gy^2)`

Далее выполняется нормализация в диапазон `0..255` и пороговая бинаризация:

`B(x,y) = 255, если G(x,y) >= T, иначе 0`

где `T = 40` (в коде: `THRESHOLD = 40`).

---

## Исходные данные
Входные файлы:

- `./input/01.png`
- `./input/02.png`
- `./input/03.png`
- `./input/04.png`
- `./input/05.png`
- `./input/06.png`
- `./input/07.png`

Для каждого изображения создаётся папка `./output/<номер>/` со всеми этапами:

- `0_collage.png`
- `1_original.png`
- `2_gray.png`
- `3_gx.png`
- `4_gy.png`
- `5_g.png`
- `6_binary.png`

---

## Результаты обработки

### Пример 1
![](./output/01/0_collage.png)
![](./output/01/1_original.png)
![](./output/01/2_gray.png)
![](./output/01/3_gx.png)
![](./output/01/4_gy.png)
![](./output/01/5_g.png)
![](./output/01/6_binary.png)

### Пример 2
![](./output/02/0_collage.png)
![](./output/02/1_original.png)
![](./output/02/2_gray.png)
![](./output/02/3_gx.png)
![](./output/02/4_gy.png)
![](./output/02/5_g.png)
![](./output/02/6_binary.png)

### Пример 3
![](./output/03/0_collage.png)
![](./output/03/1_original.png)
![](./output/03/2_gray.png)
![](./output/03/3_gx.png)
![](./output/03/4_gy.png)
![](./output/03/5_g.png)
![](./output/03/6_binary.png)

### Пример 4
![](./output/04/0_collage.png)
![](./output/04/1_original.png)
![](./output/04/2_gray.png)
![](./output/04/3_gx.png)
![](./output/04/4_gy.png)
![](./output/04/5_g.png)
![](./output/04/6_binary.png)

### Пример 5
![](./output/05/0_collage.png)
![](./output/05/1_original.png)
![](./output/05/2_gray.png)
![](./output/05/3_gx.png)
![](./output/05/4_gy.png)
![](./output/05/5_g.png)
![](./output/05/6_binary.png)

### Пример 6
![](./output/06/0_collage.png)
![](./output/06/1_original.png)
![](./output/06/2_gray.png)
![](./output/06/3_gx.png)
![](./output/06/4_gy.png)
![](./output/06/5_g.png)
![](./output/06/6_binary.png)

### Пример 7
![](./output/07/0_collage.png)
![](./output/07/1_original.png)
![](./output/07/2_gray.png)
![](./output/07/3_gx.png)
![](./output/07/4_gy.png)
![](./output/07/5_g.png)
![](./output/07/6_binary.png)

---

## Вывод
В работе реализован полный конвейер выделения границ методом Собеля:

- перевод изображения в полутоновое;
- вычисление `Gx`, `Gy`;
- расчёт модуля `G = sqrt(Gx^2 + Gy^2)`;
- пороговая бинаризация границ.

Полученные результаты показывают устойчивое выделение контуров и пригодность метода для дальнейшего анализа структуры изображений.

