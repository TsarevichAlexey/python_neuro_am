import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time as py_time

# --- Yahoo Finance ---
import yfinance as yf # Заменяем Tinkoff API

# --- Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Data Processing ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')  # Или 'Qt5Agg' в зависимости от вашей системы

# --- Конфигурация ---
TICKER = "IBM"  # <--- Тикер для Yahoo Finance
DAYS_HISTORY = 7  # <--- История за 7 дней
CANDLE_TIMEFRAME = "1m" # <--- Интервал для yfinance (1 минута)

# Параметры LSTM
N_STEPS_IN = 2167  # Используем 2 часа (120 мин) прошлых данных
N_STEPS_OUT = 60  # Предсказываем 1 час (60 мин) вперед

# Параметры обучения
SPLIT_RATIO = 0.8 # 80% данных на обучение, 20% на тест
EPOCHS = 100       # Уменьшил для скорости, можно вернуть 50
BATCH_SIZE = 32
FEATURE_COLUMN = 'Close' # <--- Название колонки в yfinance (с большой буквы)

# --- Функции ---

def get_yfinance_data(ticker, days, interval):
    """Загружает исторические данные с Yahoo Finance."""
    print(f"Загрузка данных для {ticker} за последние {days} дней (интервал {interval})...")
    try:
        # yfinance позволяет загружать 1m данные макс за 7 дней за раз
        if interval == '1m' and days > 7:
            print("Предупреждение: yfinance может ограничивать 1-минутные данные последними 7 днями. Запрашиваем 7d.")
            period = "7d"
        else:
            period = f"{days}d"

        # Используем yf.download
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False) # progress=False чтобы не было лишнего вывода

        if df is None or df.empty:
            print("Не удалось загрузить данные. Проверьте тикер и период.")
            return None

        # Обработка временной зоны - yfinance возвращает данные в TZ биржи
        # Для простоты внутри скрипта уберем информацию о TZ
        try:
            df.index = df.index.tz_localize(None)
            print(f"Временная зона удалена для обработки.")
        except TypeError:
            # Если TZ уже нет (например, для некоторых интервалов yfinance ее не ставит)
            print("Временная зона отсутствует или уже удалена.")
            pass # Игнорируем ошибку, если TZ уже нет

        df = df.sort_index() # Убедимся, что данные отсортированы по времени
        df.dropna(inplace=True) # Удаляем строки с NaN значениями, если они есть

        print(f"Загружено {len(df)} точек данных.")
        if len(df) < N_STEPS_IN + N_STEPS_OUT:
             print(f"Внимание: Загружено недостаточно данных ({len(df)}) для формирования последовательностей (требуется минимум {N_STEPS_IN + N_STEPS_OUT}).")
             print("Возможные причины: короткий период, нет торгов в запрошенное время.")
             # return None # Можно раскомментировать для прерывания

        return df

    except Exception as e:
        print(f"Произошла ошибка при загрузке данных из yfinance: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Функции prepare_data, build_lstm_model остаются почти такими же ---
def prepare_data(df, n_steps_in, n_steps_out, feature_col, split_ratio):
    """Подготавливает данные для LSTM: нормализация, создание последовательностей, разделение."""
    print("Подготовка данных для обучения...")
    min_required_len = n_steps_in + n_steps_out
    if df is None or df.empty or len(df) < min_required_len:
        print(f"Недостаточно данных для подготовки (требуется {min_required_len}, доступно {len(df) if df is not None else 0}).")
        return None, None, None, None, None, None

    # Убедимся что колонка существует
    if feature_col not in df.columns:
        print(f"Ошибка: Колонка '{feature_col}' не найдена в DataFrame. Доступные колонки: {df.columns.tolist()}")
        return None, None, None, None, None, None

    data = df[[feature_col]].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    split_idx = int(len(data) * split_ratio)

    if split_idx < n_steps_in:
         print(f"Ошибка: Слишком мало данных для обучающей выборки ({split_idx}) после разделения (требуется минимум {n_steps_in}).")
         return None, None, None, None, None, None

    scaler.fit(data[:split_idx])
    scaled_data = scaler.transform(data)

    X, y = [], []
    if len(scaled_data) < min_required_len:
         print("Ошибка: Недостаточно данных после масштабирования.")
         return None, None, None, None, None, None

    for i in range(len(scaled_data) - n_steps_in - n_steps_out + 1):
        X.append(scaled_data[i:(i + n_steps_in), 0])
        y.append(scaled_data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 0])

    if not X or not y:
        print("Ошибка: Не удалось создать последовательности X и y.")
        return None, None, None, None, None, None

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split_sequences_idx = int(len(X) * split_ratio)

    if split_sequences_idx == 0:
        print("Ошибка: Недостаточно последовательностей для обучающей выборки.")
        return None, None, None, None, None, None
    if split_sequences_idx >= len(X):
        print("Внимание: Все данные ушли на обучение, тестовой выборки не будет.")
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])
        test_indices_first_step = pd.Index([])
    else:
        X_train, X_test = X[:split_sequences_idx], X[split_sequences_idx:]
        y_train, y_test = y[:split_sequences_idx], y[split_sequences_idx:]
        num_test_samples = len(X_test)
        test_indices_start = split_sequences_idx + n_steps_in
        # Корректно берем индексы из исходного df для первого шага y_test
        test_indices_first_step = df.index[test_indices_start : test_indices_start + num_test_samples]

    print(f"Размер обучающей выборки: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Размер тестовой выборки: X_test={X_test.shape}, y_test={y_test.shape}")

    if len(X_test) == 0 :
        print("Тестовая выборка пуста.")

    return X_train, y_train, X_test, y_test, scaler, test_indices_first_step

# --- ИЗМЕНЕННАЯ функция prepare_data_diff ---
def prepare_data_diff(df, n_steps_in, n_steps_out, feature_col, split_ratio):
    """Подготавливает ДИФФЕРЕНЦИРОВАННЫЕ данные для LSTM."""
    print("Подготовка дифференцированных данных...")
    min_required_len = n_steps_in + n_steps_out + 1
    if df is None or df.empty or len(df) < min_required_len:
        print(f"Недостаточно данных для подготовки (требуется {min_required_len}, доступно {len(df) if df is not None else 0}).")
        return None, None, None, None, None, None, None # Добавил еще None

    if feature_col not in df.columns:
        print(f"Ошибка: Колонка '{feature_col}' не найдена.")
        return None, None, None, None, None, None, None

    data = df[[feature_col]].copy()
    data['diff'] = data[feature_col].diff()
    # Сохраняем исходные цены *после* точки, где взята первая разность
    original_prices_after_diff = data[feature_col].iloc[1:].values
    original_indices_after_diff = data.index[1:]
    data = data.dropna()
    diff_values = data[['diff']].values.astype(float)

    if len(diff_values) < n_steps_in + n_steps_out:
        print(f"Недостаточно данных после дифференцирования (требуется {n_steps_in + n_steps_out}, доступно {len(diff_values)}).")
        return None, None, None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    split_idx = int(len(diff_values) * split_ratio)

    if split_idx < n_steps_in:
         print(f"Ошибка: Мало данных для обучающей выборки ({split_idx}) после разделения (требуется {n_steps_in}).")
         return None, None, None, None, None, None, None

    scaler.fit(diff_values[:split_idx])
    scaled_data = scaler.transform(diff_values)

    X, y = [], []
    for i in range(len(scaled_data) - n_steps_in - n_steps_out + 1):
        X.append(scaled_data[i:(i + n_steps_in), 0])
        y.append(scaled_data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 0])

    if not X or not y:
        print("Ошибка: Не удалось создать последовательности X и y.")
        return None, None, None, None, None, None, None

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split_sequences_idx = int(len(X) * split_ratio)
    if split_sequences_idx == 0:
        print("Ошибка: Недостаточно последовательностей для обучающей выборки.")
        return None, None, None, None, None, None, None

    X_train, y_train = X[:split_sequences_idx], y[:split_sequences_idx]

    # Данные для тестовой выборки
    X_test, y_test_diff = X[split_sequences_idx:], y[split_sequences_idx:]

    # Индексы и значения для первого шага ТЕСТА в ИСХОДНЫХ данных
    original_indices_test_first_step = None
    original_values_test_first_step = None
    prices_before_test_first_step = None # Цены перед первым шагом теста

    if len(X_test) > 0:
        num_test_samples = len(X_test)
        # Начальный индекс в массиве diff_values для первого элемента X_test
        test_input_start_idx_in_diff = split_sequences_idx
        # Индекс в ИСХОДНОМ массиве цен, соответствующий ПЕРВОМУ ШАГУ ВЫХОДА y_test_diff[0]
        # Это индекс (test_input_start_idx_in_diff + n_steps_in) в массиве diff_values
        # или (test_input_start_idx_in_diff + n_steps_in + 1) в исходном df_raw
        first_output_step_original_idx_start = test_input_start_idx_in_diff + n_steps_in + 1
        original_indices_test_first_step = df.index[first_output_step_original_idx_start : first_output_step_original_idx_start + num_test_samples]
        original_values_test_first_step = df[feature_col].values[first_output_step_original_idx_start : first_output_step_original_idx_start + num_test_samples]
        # Цены прямо ПЕРЕД первым шагом выхода
        prices_before_test_first_step = df[feature_col].values[first_output_step_original_idx_start - 1 : first_output_step_original_idx_start -1 + num_test_samples]


    print(f"Размер обучающей выборки (разности): X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Размер тестовой выборки (разности): X_test={X_test.shape}, y_test={y_test_diff.shape}")

    if len(X_test) == 0 :
        print("Тестовая выборка пуста.")

    # Возвращаем все необходимые части
    return (X_train, y_train, X_test, y_test_diff, scaler,
            original_indices_test_first_step, original_values_test_first_step, prices_before_test_first_step)


# --- Функция build_lstm_model ---
def build_lstm_model(n_steps_in, n_steps_out, n_features):
    """Строит и компилирует модель LSTM."""
    print("Создание LSTM модели...")
    model = Sequential()
    # Попробуем tanh
    model.add(LSTM(units=75, activation='tanh', return_sequences=True, input_shape=(n_steps_in, n_features))) # <--- tanh
    model.add(Dropout(0.2))
    model.add(LSTM(units=75, activation='tanh')) # <--- tanh
    model.add(Dropout(0.2))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=n_steps_out))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model



def plot_predictions(actual, predicted, indices, title):
    """
    Строит график реальных и предсказанных значений (первый шаг)
    с кастомными делениями на оси Y (242-246, шаг 0.5).
    """
    # Проверки на пустые данные
    if indices is None or len(indices) == 0 \
       or actual is None or len(actual) == 0 \
       or predicted is None or len(predicted) == 0:
        print(f"Нет данных для построения графика. Индексы: {indices is not None}, actual: {actual is not None}, predicted: {predicted is not None}")
        return

    try:
        # Извлекаем данные для первого шага
        actual_plot = actual.flatten() if actual.ndim > 1 else actual
        predicted_plot = predicted.flatten() if predicted.ndim > 1 else predicted

        # Определяем длину для отрисовки
        plot_len = min(len(indices), len(actual_plot), len(predicted_plot))
        if plot_len == 0:
            print("Нет совпадающих данных для графика.")
            return

        print(f"Построение графика с {plot_len} точками...")  # Отладочное сообщение

        # Создание графика
        plt.figure(figsize=(16, 9))

        # Отрисовка линий
        plt.plot(indices[:plot_len], actual_plot[:plot_len],
                label='Реальная цена (Close)', color='blue', marker='.', linestyle='-')
        plt.plot(indices[:plot_len], predicted_plot[:plot_len],
                label='Предсказанная цена (Close)', color='red', marker='.', linestyle='--')

        # Настройки графика
        plt.title(title)
        plt.xlabel('Время')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # Автоматическое масштабирование осей Y
        y_min = min(min(actual_plot[:plot_len]), min(predicted_plot[:plot_len])) - 0.5
        y_max = max(max(actual_plot[:plot_len]), max(predicted_plot[:plot_len])) + 0.5
        plt.ylim(y_min, y_max)

        plt.tight_layout()

        # Явное отображение графика
        plt.show()
        print("График должен быть отображен")  # Отладочное сообщение

    except Exception as e:
        print(f"Ошибка при построении графика: {str(e)}")
        import traceback
        traceback.print_exc()
# --- Основной блок выполнения ---
if __name__ == "__main__":
    # 1. Загрузка данных
    df_raw = get_yfinance_data(TICKER, DAYS_HISTORY, CANDLE_TIMEFRAME)
    if df_raw is None or df_raw.empty:
        print("Не удалось получить данные. Завершение работы.")
        exit()
    # Проверка на минимальную длину для diff
    if len(df_raw) < N_STEPS_IN + N_STEPS_OUT + 1:
         print(f"Недостаточно данных ({len(df_raw)}) после загрузки для продолжения.")
         exit()

    # 2. Подготовка ДИФФЕРЕНЦИРОВАННЫХ данных
    # Распаковываем все возвращаемые значения
    result_prepare = prepare_data_diff(
            df_raw.copy(), N_STEPS_IN, N_STEPS_OUT, FEATURE_COLUMN, SPLIT_RATIO
    )

    # Проверка на None перед распаковкой
    if result_prepare is None or any(v is None for v in result_prepare if not isinstance(v, (np.ndarray, pd.Index))): # Проверяем None, кроме массивов
        print("Ошибка при подготовке данных. Завершение работы.")
        exit()

    (X_train, y_train, X_test, y_test_diff, scaler,
     original_indices_test_first_step, original_values_test_first_step,
     prices_before_test_first_step) = result_prepare


    if X_train is None or y_train is None or scaler is None: # Базовая проверка
        print("Ошибка при подготовке данных (None). Завершение работы.")
        exit()
    if X_train.shape[0] == 0:
         print("Ошибка: Обучающая выборка пуста после подготовки данных.")
         exit()

    # 3. Создание и обучение модели (на разностях)
    model = build_lstm_model(N_STEPS_IN, N_STEPS_OUT, n_features=1)
    print("Начало обучения модели...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Обучение завершено.")


    # 4. Оценка на тестовой выборке (если она есть)
    # Используем новые переменные, возвращенные из prepare_data_diff
    if X_test is not None and X_test.shape[0] > 0 and y_test_diff is not None and y_test_diff.shape[0] > 0 \
       and original_indices_test_first_step is not None and original_values_test_first_step is not None \
       and prices_before_test_first_step is not None:

        print("\n--- Оценка качества прогноза на тестовых данных (восстановленные цены) ---")
        y_pred_diff_scaled = model.predict(X_test)

        # Восстановление цен ПЕРВОГО шага
        if y_pred_diff_scaled.shape[0] == prices_before_test_first_step.shape[0]:
            # Инвертируем масштабирование для первого шага разностей
            y_pred_first_diff_unscaled = scaler.inverse_transform(y_pred_diff_scaled[:, 0].reshape(-1, 1)).flatten()

            # Восстанавливаем предсказанные цены первого шага
            pred_prices_first_step = prices_before_test_first_step + y_pred_first_diff_unscaled

            # Сравниваем с реальными значениями первого шага
            actual_prices_first_step_direct = original_values_test_first_step

            if len(actual_prices_first_step_direct) == len(pred_prices_first_step):
                # Убедимся, что оба массива 1D перед передачей в метрики
                actual_flat = actual_prices_first_step_direct.flatten()
                pred_flat = pred_prices_first_step.flatten()

                # Проверка длин после flatten (на всякий случай)
                if len(actual_flat) == len(pred_flat):

                    rmse_step1 = np.sqrt(mean_squared_error(actual_flat, pred_flat))
                    print(f"RMSE (восстановленные цены, первый шаг): {rmse_step1:.4f}")
                    epsilon = 1e-10
                    # Убедимся, что здесь тоже используем 1D массивы
                    mape_step1 = np.mean(np.abs((actual_flat - pred_flat) / (actual_flat + epsilon))) * 100
                    print(f"MAPE (восстановленные цены, первый шаг): {mape_step1:.2f}%")

                    # 5. Визуализация тестовых результатов (первый шаг)
                    if len(original_indices_test_first_step) > 0:
                         # plot_predictions ожидает 2D вход (n_samples, 1), создаем его
                         plot_predictions(
                             actual_flat.reshape(-1, 1), # Передаем как (n_samples, 1)
                             pred_flat.reshape(-1, 1),   # Передаем как (n_samples, 1)
                             original_indices_test_first_step,
                             f"Прогноз {TICKER} (восстановленные цены, первый шаг) на тесте"
                         )
                    else:
                         print("Нет индексов для графика теста.")
            else:
                print("Ошибка: Длины массивов для метрик первого шага не совпадают после восстановления.")
                print(f"Len actual direct: {len(actual_prices_first_step_direct)}")
                print(f"Len pred first step: {len(pred_prices_first_step)}")
        else:
             print("Ошибка: не совпадает кол-во предсказанных разностей и цен 'до' для теста")

    else:
        print("\nТестовая выборка пуста или не хватает данных для оценки, оценка и визуализация на тесте пропущены.")


    # 6. Сравнение прогноза и реальных данных для ПОСЛЕДНЕГО ИЗВЕСТНОГО часа
    print(f"\n--- Сравнение прогноза и реальных данных для последнего известного часа ({N_STEPS_OUT} минут, дифференцирование) ---")

    # Проверяем, достаточно ли данных в исходном df_raw
    if len(df_raw) >= N_STEPS_IN + N_STEPS_OUT + 1:
        # Берем последние N_STEPS_IN + 1 цен
        final_prices_for_input = df_raw[FEATURE_COLUMN].iloc[-(N_STEPS_IN + 1):].values
        # Реальные цены за последний час (для сравнения)
        final_actual_prices_to_compare = df_raw[FEATURE_COLUMN].iloc[-N_STEPS_OUT:].values
        final_actual_times = df_raw.index[-N_STEPS_OUT:]

        # Проверяем, что получили достаточно цен
        if final_prices_for_input.shape[0] == N_STEPS_IN + 1:
            # Считаем входные разности, УКАЗЫВАЯ ОСЬ!
            final_input_diffs = np.diff(final_prices_for_input, axis=0)

            # Проверяем, что разностей N_STEPS_IN
            if final_input_diffs.shape[0] == N_STEPS_IN:
                # Масштабируем входные разности
                final_input_diffs_scaled = scaler.transform(final_input_diffs) # reshape не нужен, если diff дал (N,1)
                final_input_diffs_scaled = final_input_diffs_scaled.reshape((1, N_STEPS_IN, 1)) # Reshape для LSTM

                # Делаем прогноз разностей
                print("Выполнение прогноза разностей для последнего часа...")
                final_pred_diff_scaled = model.predict(final_input_diffs_scaled)

                # Обратно масштабируем прогноз разностей
                final_inv_pred_diff = scaler.inverse_transform(final_pred_diff_scaled.reshape(-1, 1)).flatten()

                # Восстанавливаем прогноз цен
                last_price_before_final_hour = final_prices_for_input[-1, 0] # Берем скалярное значение
                final_predicted_prices = np.zeros_like(final_inv_pred_diff)
                current_price = last_price_before_final_hour
                for i in range(len(final_inv_pred_diff)):
                    current_price += final_inv_pred_diff[i]
                    final_predicted_prices[i] = current_price

                # Сравнение
                if len(final_predicted_prices) == len(final_actual_prices_to_compare) == len(final_actual_times):
                    print(f"Сравнение для последних {N_STEPS_OUT} временных точек (последний доступный час):")
                    print("-" * 65)
                    print(f"{'Время':<25} {'Прогноз (восст.)':<18} {'Реальность':<18}")
                    print("-" * 65)
                    valid_preds_last = []
                    valid_actuals_last = []
                    for t, p, a in zip(final_actual_times, final_predicted_prices, final_actual_prices_to_compare):
                        time_str = t.strftime('%Y-%m-%d %H:%M:%S')
                        pred_str = f"{float(p):.4f}"
                        act_str = f"{float(a):.4f}"
                        print(f"{time_str:<25} {pred_str:<18} {act_str:<18}")
                        valid_preds_last.append(float(p))
                        valid_actuals_last.append(float(a))
                    print("-" * 65)
                    if valid_preds_last and valid_actuals_last:
                        rmse_last = np.sqrt(mean_squared_error(valid_actuals_last, valid_preds_last))
                        epsilon = 1e-10
                        mape_last = np.mean(np.abs((np.array(valid_actuals_last) - np.array(valid_preds_last)) / (np.array(valid_actuals_last) + epsilon))) * 100
                        print(f"RMSE для прогноза последнего часа (восст.): {rmse_last:.4f}")
                        print(f"MAPE для прогноза последнего часа (восст.): {mape_last:.2f}%")
                else:
                    print("Ошибка: Длины массивов для сравнения последнего часа не совпадают.")
                print(f"Длина прогноза цен: {len(final_predicted_prices)}, Длина реальных цен: {len(final_actual_prices_to_compare)}, Длина времен: {len(final_actual_times)}")
            else:
                 print(f"Ошибка: Получено {final_input_diffs.shape[0]} разностей для последнего часа, ожидалось {N_STEPS_IN}.")
        else:
             print(f"Ошибка: Не удалось извлечь {N_STEPS_IN + 1} цен для вычисления разностей последнего часа (извлечено {final_prices_for_input.shape[0]}).")
    else:
        print("Недостаточно данных в df_raw для выполнения прогноза на последнем часе.")

# Конец программы
