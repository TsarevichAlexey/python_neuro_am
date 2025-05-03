import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time as py_time

# --- Tinkoff Invest API --- # ! ИЗМЕНЕНО !
from tinkoff.invest import Client, CandleInterval, RequestError
from tinkoff.invest.utils import now, quotation_to_decimal # Добавлен quotation_to_decimal

# --- Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Data Processing ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- Конфигурация --- # ! ИЗМЕНЕНО !
# !!! ВАЖНО: Замените на ваш реальный токен API !!!
TINKOFF_API_TOKEN = "t.He-UEc5Mfe_E2w9CVSlZsCrqyoFacn52Yut_rwdsK8EclvfaxLlju33-26Z2cW22w0dDKNK0cNpxsyXLSGZv-g" # <--- ЗАМЕНИТЕ ВАШИМ ТОКЕНОМ
INSTRUMENT_FIGI = "BBG004730N88"  # <--- FIGI Сбербанка (или другого инструмента)
DAYS_HISTORY = 1  # <--- История за 1 день
CANDLE_TIMEFRAME = CandleInterval.CANDLE_INTERVAL_1_MIN # <--- Интервал Tinkoff (1 минута)

# Параметры LSTM
# !!! ВАЖНО: N_STEPS_IN должно быть меньше кол-ва свечей за день !!!
N_STEPS_IN = 940  # <--- Возвращено к 2 часам (120 мин) прошлых данных
N_STEPS_OUT = 60  # Предсказываем 1 час (60 мин) вперед

# Параметры обучения
SPLIT_RATIO = 0.8 # 80% данных на обучение, 20% на тест
EPOCHS = 100
BATCH_SIZE = 32
FEATURE_COLUMN = 'close' # <--- Название колонки создаваемого DataFrame (строчная)

# --- Функции ---

def decimal_to_float(quotation):
    """Конвертирует Quotation в float."""
    # Обработка случая, когда quotation может быть None или не иметь нужных атрибутов
    if quotation and hasattr(quotation, 'units') and hasattr(quotation, 'nano'):
        try:
             # Используем стандартную утилиту, если она доступна и работает
             return float(quotation_to_decimal(quotation))
        except Exception:
             # Запасной вариант, если quotation_to_decimal дает сбой
             return float(f"{quotation.units}.{quotation.nano:09d}")
    return np.nan # Возвращаем NaN, если конвертация невозможна


# ! НОВАЯ функция get_tinkoff_candles !
def get_tinkoff_candles(token, figi, days, interval):
    """Загружает исторические данные свечей с Tinkoff API v2."""
    print(f"Загрузка данных для FIGI: {figi} за последние {days} дней ({interval.name})...")
    all_candles_df = pd.DataFrame()
    try:
        with Client(token) as client:
            end_dt = now() # Используем now() из tinkoff utils
            start_dt = end_dt - timedelta(days=days)

            print(f"Запрос свечей с {start_dt.strftime('%Y-%m-%d %H:%M')} по {end_dt.strftime('%Y-%m-%d %H:%M')}")

            candles_list = []
            # Используем итератор для обработки
            for candle in client.get_all_candles(
                figi=figi,
                from_=start_dt,
                to=end_dt,
                interval=interval
            ):
                candles_list.append(candle)
            py_time.sleep(0.1) # Небольшая пауза

            if not candles_list:
                print("Не удалось загрузить данные. Проверьте FIGI, период и наличие торгов.")
                return None

            # Преобразование в DataFrame
            df = pd.DataFrame([{
                'time': pd.Timestamp(c.time).tz_localize(None), # Убираем tz_info
                'open': decimal_to_float(c.open),
                'high': decimal_to_float(c.high),
                'low': decimal_to_float(c.low),
                'close': decimal_to_float(c.close), # Используем 'close'
                'volume': c.volume,
                'is_complete': c.is_complete
            } for c in candles_list])

            df = df.set_index('time')
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.dropna(subset=[FEATURE_COLUMN], inplace=True) # Удаляем строки, где цена NaN

            print(f"Загружено {len(df)} свечей.")
            # Проверка на достаточность данных для diff
            if len(df) < N_STEPS_IN + N_STEPS_OUT + 1:
                 print(f"Внимание: Загружено недостаточно данных ({len(df)}) для формирования последовательностей (требуется минимум {N_STEPS_IN + N_STEPS_OUT + 1}).")
                 print("Возможные причины: выходной день, короткий торговый день, нет данных для инструмента.")

            return df

    except RequestError as e:
        print(f"Ошибка API Tinkoff: {e}")
        if hasattr(e, 'details') and e.details:
             print(f"Details (from gRPC): {e.details}")
        if hasattr(e, 'metadata') and e.metadata:
             print(f"Metadata: {e.metadata}")
        return None
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Функция prepare_data_diff --- (остается прежней, работает с DataFrame)
def prepare_data_diff(df, n_steps_in, n_steps_out, feature_col, split_ratio):
    """Подготавливает ДИФФЕРЕНЦИРОВАННЫЕ данные для LSTM."""
    print("Подготовка дифференцированных данных...")
    min_required_len = n_steps_in + n_steps_out + 1
    if df is None or df.empty or len(df) < min_required_len:
        print(f"Недостаточно данных для подготовки (требуется {min_required_len}, доступно {len(df) if df is not None else 0}).")
        return None, None, None, None, None, None, None, None # 8 return values

    if feature_col not in df.columns:
        print(f"Ошибка: Колонка '{feature_col}' не найдена.")
        return None, None, None, None, None, None, None, None

    data = df[[feature_col]].copy()
    data['diff'] = data[feature_col].diff()
    original_prices_after_diff = data[feature_col].iloc[1:].values # Not used later, but kept for consistency
    original_indices_after_diff = data.index[1:] # Not used later, but kept for consistency
    data = data.dropna()
    diff_values = data[['diff']].values.astype(float)

    if len(diff_values) < n_steps_in + n_steps_out:
        print(f"Недостаточно данных после дифференцирования (требуется {n_steps_in + n_steps_out}, доступно {len(diff_values)}).")
        return None, None, None, None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    split_idx = int(len(diff_values) * split_ratio)

    if split_idx < n_steps_in:
         print(f"Ошибка: Мало данных для обучающей выборки ({split_idx}) после разделения (требуется {n_steps_in}).")
         return None, None, None, None, None, None, None, None

    scaler.fit(diff_values[:split_idx])
    scaled_data = scaler.transform(diff_values)

    X, y = [], []
    for i in range(len(scaled_data) - n_steps_in - n_steps_out + 1):
        X.append(scaled_data[i:(i + n_steps_in), 0])
        y.append(scaled_data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 0])

    if not X or not y:
        print("Ошибка: Не удалось создать последовательности X и y.")
        return None, None, None, None, None, None, None, None

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split_sequences_idx = int(len(X) * split_ratio)
    if split_sequences_idx == 0:
        print("Ошибка: Недостаточно последовательностей для обучающей выборки.")
        return None, None, None, None, None, None, None, None

    X_train, y_train = X[:split_sequences_idx], y[:split_sequences_idx]
    X_test, y_test_diff = X[split_sequences_idx:], y[split_sequences_idx:]

    original_indices_test_first_step = None
    original_values_test_first_step = None
    prices_before_test_first_step = None

    if len(X_test) > 0:
        num_test_samples = len(X_test)
        test_input_start_idx_in_diff = split_sequences_idx
        first_output_step_original_idx_start = test_input_start_idx_in_diff + n_steps_in + 1

        # Check bounds before slicing original DataFrame 'df'
        if first_output_step_original_idx_start + num_test_samples <= len(df):
             original_indices_test_first_step = df.index[first_output_step_original_idx_start : first_output_step_original_idx_start + num_test_samples]
             original_values_test_first_step = df[feature_col].values[first_output_step_original_idx_start : first_output_step_original_idx_start + num_test_samples]
             prices_before_test_first_step = df[feature_col].values[first_output_step_original_idx_start - 1 : first_output_step_original_idx_start -1 + num_test_samples]
        else:
            print("Предупреждение: Не хватает данных в исходном DataFrame для извлечения полных тестовых данных (индексы/значения). Тест может быть неполным.")
            # Try slicing what's available
            available_len = len(df) - first_output_step_original_idx_start
            if available_len > 0:
                 original_indices_test_first_step = df.index[first_output_step_original_idx_start : first_output_step_original_idx_start + available_len]
                 original_values_test_first_step = df[feature_col].values[first_output_step_original_idx_start : first_output_step_original_idx_start + available_len]
                 prices_before_test_first_step = df[feature_col].values[first_output_step_original_idx_start - 1 : first_output_step_original_idx_start -1 + available_len]
            # If even that fails, variables remain None

    print(f"Размер обучающей выборки (разности): X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Размер тестовой выборки (разности): X_test={X_test.shape}, y_test={y_test_diff.shape}")

    if len(X_test) == 0 :
        print("Тестовая выборка пуста.")

    return (X_train, y_train, X_test, y_test_diff, scaler,
            original_indices_test_first_step, original_values_test_first_step, prices_before_test_first_step)


# --- Функция build_lstm_model --- (остается прежней)
def build_lstm_model(n_steps_in, n_steps_out, n_features):
    print("Создание LSTM модели...")
    model = Sequential()
    model.add(LSTM(units=75, activation='tanh', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=75, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=n_steps_out))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


# --- Функция plot_predictions --- # ! ИЗМЕНЕНО !
def plot_predictions(actual, predicted, indices, title):
    """Строит график реальных и предсказанных значений (первый шаг)."""
    if indices is None or len(indices) == 0 \
       or actual is None or actual.shape[0] == 0 \
       or predicted is None or predicted.shape[0] == 0:
        print("Нет данных для построения графика.")
        return

    actual_plot = actual[:, 0]
    predicted_plot = predicted[:, 0]

    plot_len = min(len(indices), len(actual_plot), len(predicted_plot))
    if plot_len == 0 :
        print("Нет совпадающих данных для графика.")
        return

    plt.figure(figsize=(16, 9))
    plt.plot(indices[:plot_len], actual_plot[:plot_len], label='Реальная цена (close)', color='blue', marker='.', linestyle='-') # Изменено на close
    plt.plot(indices[:plot_len], predicted_plot[:plot_len], label='Предсказанная цена (close)', color='red', marker='.', linestyle='--') # Изменено на close

    # --- Убраны жестко заданные Yticks для IBM ---
    # plt.yticks(np.arange(242, 246.5, 0.5)) # Закомментировано
    # plt.ylim(241.5, 246.5) # Закомментировано
    # Matplotlib подберет деления автоматически для Сбербанка
    # ---

    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Основной блок выполнения ---
if __name__ == "__main__":
    # 1. Загрузка данных # ! ИЗМЕНЕНО !
    df_raw = get_tinkoff_candles(TINKOFF_API_TOKEN, INSTRUMENT_FIGI, DAYS_HISTORY, CANDLE_TIMEFRAME)

    if df_raw is None or df_raw.empty:
        print("Не удалось получить данные. Завершение работы.")
        exit()
    # Проверка на минимальную длину для diff
    if len(df_raw) < N_STEPS_IN + N_STEPS_OUT + 1:
         print(f"Недостаточно данных ({len(df_raw)}) после загрузки для продолжения (нужно {N_STEPS_IN + N_STEPS_OUT + 1}).")
         exit()

    # 2. Подготовка ДИФФЕРЕНЦИРОВАННЫХ данных # ! ИЗМЕНЕНО !
    result_prepare = prepare_data_diff(
            df_raw.copy(), N_STEPS_IN, N_STEPS_OUT, FEATURE_COLUMN, SPLIT_RATIO # Используем 'close'
    )

    if result_prepare is None:
        print("Ошибка при подготовке данных (result_prepare is None). Завершение работы.")
        exit()
    # Check if any essential component is None (excluding potentially empty test arrays/indices)
    if result_prepare[0] is None or result_prepare[1] is None or result_prepare[4] is None: # Check X_train, y_train, scaler
        print("Ошибка при подготовке данных (X_train/y_train/scaler is None). Завершение работы.")
        exit()

    (X_train, y_train, X_test, y_test_diff, scaler,
     original_indices_test_first_step, original_values_test_first_step,
     prices_before_test_first_step) = result_prepare

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
    if X_test is not None and X_test.shape[0] > 0 and y_test_diff is not None and y_test_diff.shape[0] > 0 \
       and original_indices_test_first_step is not None and original_values_test_first_step is not None \
       and prices_before_test_first_step is not None:

        print("\n--- Оценка качества прогноза на тестовых данных (восстановленные цены) ---")
        y_pred_diff_scaled = model.predict(X_test)

        if y_pred_diff_scaled.shape[0] == prices_before_test_first_step.shape[0]:
            y_pred_first_diff_unscaled = scaler.inverse_transform(y_pred_diff_scaled[:, 0].reshape(-1, 1)).flatten()
            pred_prices_first_step = prices_before_test_first_step + y_pred_first_diff_unscaled
            actual_prices_first_step_direct = original_values_test_first_step

            # Ensure comparison arrays have the same final length after potential slicing in prepare_data
            min_len = min(len(actual_prices_first_step_direct), len(pred_prices_first_step))
            if min_len > 0 :
                actual_flat = actual_prices_first_step_direct[:min_len].flatten()
                pred_flat = pred_prices_first_step[:min_len].flatten()
                indices_flat = original_indices_test_first_step[:min_len]

                if len(actual_flat) == len(pred_flat):
                    rmse_step1 = np.sqrt(mean_squared_error(actual_flat, pred_flat))
                    print(f"RMSE (восстановленные цены, первый шаг): {rmse_step1:.4f}")
                    epsilon = 1e-10
                    mape_step1 = np.mean(np.abs((actual_flat - pred_flat) / (actual_flat + epsilon))) * 100
                    print(f"MAPE (восстановленные цены, первый шаг): {mape_step1:.2f}%")

                    # 5. Визуализация тестовых результатов (первый шаг)
                    if len(indices_flat) > 0:
                         plot_predictions(
                             actual_flat.reshape(-1, 1),
                             pred_flat.reshape(-1, 1),
                             indices_flat,
                             f"Прогноз {INSTRUMENT_FIGI} (восстановленные цены, первый шаг) на тесте" # ! Используем FIGI
                         )
                    else:
                         print("Нет индексов для графика теста.")
                else:
                    print("Ошибка: Длины массивов не совпадают ПОСЛЕ flatten() и min_len.")
            else:
                 print("Ошибка: Недостаточно совпадающих данных для расчета метрик первого шага.")
        else:
             print("Ошибка: не совпадает кол-во предсказанных разностей и цен 'до' для теста")
    else:
        print("\nТестовая выборка пуста или не хватает данных для оценки, оценка и визуализация на тесте пропущены.")


    # 6. Сравнение прогноза и реальных данных для ПОСЛЕДНЕГО ИЗВЕСТНОГО часа
    print(f"\n--- Сравнение прогноза и реальных данных для последнего известного часа ({N_STEPS_OUT} минут, дифференцирование) ---")

    if len(df_raw) >= N_STEPS_IN + N_STEPS_OUT + 1:
        final_prices_for_input = df_raw[FEATURE_COLUMN].iloc[-(N_STEPS_IN + 1):].values
        final_actual_prices_to_compare = df_raw[FEATURE_COLUMN].iloc[-N_STEPS_OUT:].values
        final_actual_times = df_raw.index[-N_STEPS_OUT:]
                # 1. Reshape 1D diffs to 2D for the scaler
        final_input_diffs_2d = final_input_diffs.reshape(-1, 1)

                # 2. Scale the 2D array
        final_input_diffs_scaled = scaler.transform(final_input_diffs_2d)
        if final_prices_for_input.shape[0] == N_STEPS_IN + 1:
            final_input_diffs = np.diff(final_prices_for_input, axis=0)

            if final_input_diffs.shape[0] == N_STEPS_IN:
                # 1. Reshape 1D diffs to 2D for the scaler
                final_input_diffs_2d = final_input_diffs.reshape(-1, 1)

                # 2. Scale the 2D array
                final_input_diffs_scaled = scaler.transform(final_input_diffs_2d)

                final_input_diffs_scaled = final_input_diffs_scaled.reshape((1, N_STEPS_IN, 1))

                print("Выполнение прогноза разностей для последнего часа...")
                final_pred_diff_scaled = model.predict(final_input_diffs_scaled)
                final_inv_pred_diff = scaler.inverse_transform(final_pred_diff_scaled.reshape(-1, 1)).flatten()

                last_price_before_ficandnal_hour = final_prices_for_input[-1, 0]
                final_predicted_prices = np.zeros_like(final_inv_pred_diff)
                current_price = last_price_before_final_hour
                for i in range(len(final_inv_pred_diff)):
                    current_price += final_inv_pred_diff[i]
                    final_predicted_prices[i] = current_price

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
