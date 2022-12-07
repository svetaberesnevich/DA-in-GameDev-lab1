# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе №1 выполнил:
- Шамонин Филипп Николаевич
- РИ-210941
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
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
- Для Python в отчете привести скриншоты с демонстрацией сохранения документа google.colab на свой диск с запуском программы, выводящей сообщение Hello World.

![1](https://user-images.githubusercontent.com/120024952/206180979-41869665-ad6e-4fb9-890a-5a7e3fe25775.png)

![2](https://user-images.githubusercontent.com/120024952/206181003-63db7ca6-a173-4a9e-a505-a9657bfbecc6.png)

- Для Unity в отчете привести скриншоты вывода сообщения Hello World в консоль.
 
 ![UnityHelloWorld](https://user-images.githubusercontent.com/103362219/192449702-89f297d9-abda-448f-9eeb-9d1246a7b736.png)

## Задание 2. В разделе "ход работы" пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.
Ход работы:
1. Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

plt.scatter(x,y)

```
![image](https://user-images.githubusercontent.com/120024952/206209214-76c4161e-f9a4-4d28-b75e-db65075f8b66.png)


2. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
def model(a,b,x):
  return a*x+b

def loss_function(a,b,x,y):
    num = len(x)
    prediction=model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()
    
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    da = (1.0 /num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * ((prediction - y). sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b 
def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b
```
![image](https://user-images.githubusercontent.com/120024952/206209408-1dec3c9f-8eec-4d13-a610-f328d8230df7.png)
3. Начать итерацию

  Шаг 1. Инициализация и модель итерактивной оптимизации
```py
a = np. random. rand(1)
print(a)
b = np. random. rand(1)
print(b)
Lr = 0.000001
 
a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206209524-f7136830-5b89-48a1-9f47-dbed30695ee6.png)
Шаг 2. На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
```py
a, b = iterate(a, b, x, y, 2)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206209634-0f09f00d-530a-482c-84a5-3d84ca6bf43b.png)

Шаг 3. Третья итерация показывает значения параметров, значения потерь и эффекты визуализации после итерации
```py
a, b = iterate(a, b, x, y, 3)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206209716-6280d275-b90b-4b1d-830b-80f7c6dc6c11.png)

Шаг 4. На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
```py
a, b = iterate(a, b, x, y, 4)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206209835-1a27dcf6-704b-486a-b685-3fb33fe34294.png)

Шаг 5. Пятая итерация показывает значения параметров, значения потерь и эффекты визуализации после итерации
```py
a, b = iterate(a, b, x, y, 5)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206209963-ba893076-1195-4edc-9c7c-5ff50db70263.png)

Шаг 6. 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
```py
a, b = iterate(a, b, x, y, 10000)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt. scatter(x, y)
plt. plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/120024952/206210043-3a77281f-a2c4-40b1-8cf6-d637a4f9c714.png)


## Задание 3
### Какова роль параметра Lr? 

Параметр Lr определяет, как резко будет возрастать, либо же убывать график в зависимости от его значения.

![image](https://user-images.githubusercontent.com/120024952/206210191-01ced55c-feb0-44b4-aafb-5b854af4b130.png)
![image](https://user-images.githubusercontent.com/120024952/206210268-21cd9819-38f9-4974-9dc0-7db29b5bac16.png)
![image](https://user-images.githubusercontent.com/120024952/206210365-fac114ad-3c02-4c53-9d4a-67d0a90833a9.png)
![image](https://user-images.githubusercontent.com/120024952/206210436-0f8d472c-85e5-447a-85cd-95d82bcf7312.png)

### Должна ли величина loss стремиться к нулю при изменении исходных данных? 
Нет, не должна, это видно по построенному графику:

![image](https://user-images.githubusercontent.com/120024952/206210587-d8b248fe-91eb-49a7-a9b1-d8a2adff4815.png)
![image](https://user-images.githubusercontent.com/120024952/206210694-5feef30b-80f8-436f-bee8-2ad0d9b9249f.png)

## Выводы

В ходе проделанной работы я ознакомился с основными операторами зыка Python на примере реализации линейной регрессии, получил опыт работы с анализом данных, а также поработал в таких средах разработки, как: PyCharm, Anaconda и Unity. В целом, я усвоил для себя много нового и интересного.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
