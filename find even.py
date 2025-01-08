import numpy as np


# Функция для поиска всех чисел, которые встречаются в массиве чётное количество раз (задача №15)
def even_digits(start_list: list) -> np.array:
    # Пробуем провести рассчёты с полученным входным параметром
    try:
        # Создаём нужный нам array чисел
        arr = np.array(start_list)
        # Создаём array уникальных чисел > 0, соответствующий индексу в array
        counts = np.bincount(arr)
        # Создаём новый array на основе начального, где каждый элемент заменён на его количество в начальном списке
        res = counts[arr]
        # Оставляем только значения, которые встречают чётное количество раз
        return arr[res % 2 == 0]
    # Ловим ошибку из-за недопустимого значения
    except ValueError:
        # Выдаём в терминал сообщение о наличии ошибки: недопустимое значение
        print("Недопустимое значение")


# Функция для реализации тестов
def test() -> None:
    assert(list(even_digits([1, 1, 2, 3, 3, 3])) == [1, 1])
    print("First test successfully passed")
    assert (list(even_digits([5, 1, 2, 3, 3, 5])) == [5, 3, 3, 5])
    print("Second test successfully passed")
    assert (list(even_digits([8, 8, 8, 8, 8, 8])) == [8, 8, 8, 8, 8, 8])
    print("Third test successfully passed")
    assert (list(even_digits([1])) == [])
    print("Fourth test successfully passed")


test()
