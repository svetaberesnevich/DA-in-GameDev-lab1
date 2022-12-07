# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе №5 выполнил:
- Берсенева Светлана Александровна
- ФО-210005
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы: освоить применение направления науки о данных в играх и интерактивных приложениях.

- Я выполнил все методические указания для запуска ML-агента.

![4](https://user-images.githubusercontent.com/103362219/205246237-be60a6f3-a3a0-4404-9969-d855cab5619e.png)

![1](https://user-images.githubusercontent.com/103362219/205246318-3bca0ee2-988d-472e-af57-8b3b54c5a4d4.JPG)

![3](https://user-images.githubusercontent.com/103362219/205246342-b65b819d-35c9-435d-ae14-afe0c453ba8f.png)

![2](https://user-images.githubusercontent.com/103362219/205246365-4c9a7e22-6609-4196-bd4c-8e20aff484d7.JPG)

## Задание 1. Изменить параметры файла yaml-агента и определить какие параметры и как влияют на обучение модели.

### Ход работы:

- Использовал веб-приложение"TensorBoard"

![7](https://user-images.githubusercontent.com/103362219/205248509-845109da-d24d-465f-8669-591315d2aaa0.JPG)

![6](https://user-images.githubusercontent.com/103362219/205248518-a0ed2ff6-dd4d-4d70-baa9-dbc06f318cd1.JPG)

![5](https://user-images.githubusercontent.com/103362219/205248530-c3e68f36-a3bb-4019-9c4f-f7d792f29a8d.JPG)

![8](https://user-images.githubusercontent.com/103362219/205248544-df37d93a-acec-4e2a-bd56-810e6eb2a2a6.JPG)

![9](https://user-images.githubusercontent.com/103362219/205248557-a3929827-981f-47c0-bfba-16646b84245a.JPG)

### Из приведённых графиков можно сделать вывод, что параметры "checkpoint_interval", "strength", "max_steps" и "gamma" заметно меняют исход обучения модели. Наибольшее влияние на обучение модели оказывает параметр "strength".

## Задание 2. Описать результаты, выведенные в TensorBoard.

Environment: этот раздел показывает, как агент проявляет себя в виртуальной среде.

Losses: в этом разделе показаны графики, представляющие вычисленные потери или затраты для политики и значения.

Value Loss: это средняя потеря функции значения.

Cumulative Reward: это общее вознаграждение, которое максимизирует агент. Чаще всего, оно увеличивается, но по некоторым причинам оно может и уменьшаться.

Episode Length: чем короче сами эпизоды, тем дольше обучение.

Policy Loss: этот график определяет величину изменения политики со временем.

## Выводы

### В ходе проделанной работы я смогла разобраться с материалом и обучить ML-agent'a. Также я поработала в веб-приложении "TensorBoard" для визуализации данных в эксперименте. Поняла, что экономика в играх - это очень важная вещь, требующая тщательной проработки.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |
