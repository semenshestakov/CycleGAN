# CycleGAN


### О данных
Данные храняться в дириктории data, а именно monet и photo \
monet - это разные картины, photos - это фотографии реального мира

Все изображения формата jpg и размером (256, 256, 3)

data/extract_data - храниться класс для работы с данными \
data/data.csv информация, о файлах (их название, расположение)
data/monet - изображения картин 
data/photo - изображения фотографий реального мира


## Models
models.py находятся функции и классы нейронных сетей, а именно:

### get_generator_on_vgg16
В данном генераторе я использовал свёрточные слои VGG16, заморозил их, чтобы обучать меньше параметров.

![get_generator_on_vgg16](https://github.com/semenshestakov/CycleGAN/assets/94396766/34a8d319-d40c-44aa-b82d-3ffa3138631c)

### discriminator
![discriminator](https://github.com/semenshestakov/CycleGAN/assets/94396766/6c4bfe6d-81eb-4ae9-ba99-5ba373b78c42)

### Математика и представления CycleGAN
![telegram-cloud-photo-size-2-5361755576993958339-y](https://github.com/semenshestakov/CycleGAN/assets/94396766/478f15cb-c709-489c-9926-bf16a4b2bda9)

#### Схема
<img width="290" alt="Screenshot 2023-08-01 at 12 22 07" src="https://github.com/semenshestakov/CycleGAN/assets/94396766/19ec1c7b-fab1-40dd-a055-524fa4add958">


## Обучение, гипотезы, идеи 
Обучение я проводил в Сolab. Обучал на GPU вышло примерно 170 минут на обучение 100 мини эпох.
Мини эпоха - это 1/10 полной эпохи, было это создано ради более детального наблюдения, как обучается сеть.

Первое время генератор сводил любой результат к черному экрану. Проблема оказалось в discriminator, так как первый вариант сводил размерность к (batch, 1), потом я нашел решение, оказалось в том что надо сводить например к (batch, size, size, 1), где я сводил размерность к (16,16,1). Идея заключаться в том, что мы рассматриваем изображение кусочками. Такой подход решил проблему черного экрана. Вторая проблема описана в гипотезе 2.

#### Гипотеза 1.
Слои VGG16 позволили не только сходиться быстрее, но и находить более четки границы обектов.


#### Гипотеза 2.
Instance Normalization(IN) - позволяет избавиться от сетки, которая появлялась при обучении без IN.
Пример с сеткой, это первый хороший результат, который выдала НС без Instance Normalization. Сходимость к более-менее хорошему результату была очень быстро. Правда клетка мешала. 
![telegram-cloud-photo-size-2-5368758315011656341-y](https://github.com/semenshestakov/CycleGAN/assets/94396766/a621a5a0-b033-41b0-a85a-83221198f629)

Из-за ограничения в мощности Colab падал с ошибкой, если использовать Instance Normalization после concatenate. Ошибка связана с ограничением в оперативной памяти. 
Но в финальном варианте, когда я аккуратно подобрал параметры м расположение IN – это мне позволило мне обучать модель.

### Результат после каждой мини эпохи.


https://github.com/semenshestakov/CycleGAN/assets/94396766/7f5edbad-5e0a-42fd-a521-fddbdc52bd84


![1df4ff0436](https://github.com/semenshestakov/CycleGAN/assets/94396766/3f858a18-6f45-4ec2-a25b-3fc70351a283)
![1e90517e22](https://github.com/semenshestakov/CycleGAN/assets/94396766/24c5e197-5797-4e55-808a-ac8b4135a1d8)
![1f006ed5eb](https://github.com/semenshestakov/CycleGAN/assets/94396766/5fa3120a-47cc-4855-9f3e-5db852a1970f)
![5bb7d3f1e2](https://github.com/semenshestakov/CycleGAN/assets/94396766/c71950a4-968e-4e7a-a537-67d019984a02)
![6b47c83dad](https://github.com/semenshestakov/CycleGAN/assets/94396766/00df1a0f-619d-40be-bd94-6a5df47149f8)
![6e9f6935ce](https://github.com/semenshestakov/CycleGAN/assets/94396766/9f73d859-810d-4d54-ac49-001d13fd1866)
![8af0546eb3](https://github.com/semenshestakov/CycleGAN/assets/94396766/ec5b8398-e7d0-4e8c-a1d5-7c0fc682af7c)
![daf0f4bdc1](https://github.com/semenshestakov/CycleGAN/assets/94396766/e579ac3a-5075-45bc-8080-11edbc87ba79)
![df571c611d](https://github.com/semenshestakov/CycleGAN/assets/94396766/c8530dc4-b167-4b90-94b3-8294082a14e8)


### Большое результатов в data/result

Есть много очень хороших и есть много очень плохих. Во время обучения с IN - сразу стало заметно, что сетка почти пропала.
#### Гипотеза 2 - ПОДТВЕРДИЛАСЬ

При инициализации весов и при использовании обученных параметров VGG16 сразу стало понятно, что очертания объектов видны, а как только убрать обученные веста, то сразу появлялся хаус при инициализации. Пример на 0 эпохе(до обучения) при обученных VGG16 весах и без IN:
![image_at_epochNone](https://github.com/semenshestakov/CycleGAN/assets/94396766/ba2a9df3-1db7-48cd-b1d0-abb84c87244c)
#### Гипотеза 1 - ПОДТВЕРДИЛАСЬ




















































