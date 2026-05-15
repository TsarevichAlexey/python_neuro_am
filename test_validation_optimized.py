#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tinkoff.invest import Client, CandleInterval, RequestError
from tinkoff.invest.utils import now, quotation_to_decimal

# Убедитесь, что у вас есть файл config.py с вашим токеном
# Пример содержимого config.py: TOKEN = "ваш_реальный_токен_здесь"
try:
    from config import TOKEN
except ImportError:
    TOKEN = "YOUR_TINKOFF_API_TOKEN" # Замените на ваш токен, если нет config.py
    print("WARN: config.py not found or TOKEN not defined. Using placeholder token.")

TINKOFF_API_TOKEN = TOKEN

# --- Configuration ---
INSTRUMENT_FIGI = "BBG004730N88" # FIGI Сбербанка (SBER) как пример
DAYS_HISTORY = 30                # Количество дней для загрузки исторических данных
CANDLE_TIMEFRAME = CandleInterval.CANDLE_INTERVAL_1_MIN # Таймфрейм свечей
MA_WINDOW = 20                   # Окно для скользящей средней
N_STEPS_IN = 60                  # Количество временных шагов на входе модели (например, 60 минут)
N_STEPS_OUT = 30                 # Количество временных шагов для предсказания (например, 30 минут)
PRICE_COLUMN = 'close'           # Колонка с ценой для предсказания
N_FEATURES = 2                   # Количество признаков (цена + MA)
FORECAST_HORIZON_MINUTES = 30    # Горизонт для независимого прогноза в минутах

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

#%%
def get_tinkoff_candles(token, figi, days, interval):
    """Загружает исторические данные по свечам с Tinkoff API."""
    print(f"Загрузка {days} дней свечей ({interval}) для FIGI {figi}...")
    if token == "YOUR_TINKOFF_API_TOKEN":
        print("Ошибка: Не указан реальный API токен Tinkoff. Используется заглушка.")
        # Создаем фейковые данные для демонстрации, если нет токена
        print("Создание фейковых данных для демонстрации...")
        date_rng = pd.date_range(end=datetime.now(), periods=days * 24 * 60, freq='T') # Примерно минут в днях
        data = np.random.rand(len(date_rng)) * 100 + 200 # Случайные цены
        df = pd.DataFrame(data, index=date_rng, columns=['close'])
        df['open'] = df['close'] - np.random.rand(len(date_rng)) * 2
        df['high'] = df['close'] + np.random.rand(len(date_rng)) * 2
        df['low'] = df['open'] - np.random.rand(len(date_rng)) * 2
        df['volume'] = np.random.randint(100, 1000, size=len(date_rng))
        df.index.name = 'time'
        df = df.dropna()
        print(f"Создано {len(df)} фейковых свечей.")
        return df

    with Client(token) as client:
        end_dt = now()
        start_dt = end_dt - timedelta(days=days)

        candles_data = []
        try:
            for candle in client.get_all_candles(figi=figi, from_=start_dt, to=end_dt, interval=interval):
                candles_data.append({
                    'time': pd.Timestamp(candle.time).tz_localize(None), # Убираем информацию о часовом поясе
                    'open': float(quotation_to_decimal(candle.open)),
                    'high': float(quotation_to_decimal(candle.high)),
                    'low': float(quotation_to_decimal(candle.low)),
                    'close': float(quotation_to_decimal(candle.close)),
                    'volume': candle.volume
                })
        except RequestError as e:
            print(f"Ошибка API Tinkoff: {e}")
            return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки

        if not candles_data:
            print("Данные не загружены. Проверьте FIGI, токен и период.")
            return pd.DataFrame()

        df = pd.DataFrame(candles_data)
        df = df.set_index('time').sort_index()
        df.dropna(inplace=True)

        print(f"Загружено {len(df)} свечей.")
        if not df.empty:
            print("Пример загруженных данных (первые и последние 2 строки):")
            print(pd.concat([df.head(2), df.tail(2)]))
        return df

#%%
def add_ma_and_scale_data(df, price_col, ma_window):
    """Добавляет скользящую среднюю и масштабирует данные."""
    if price_col not in df.columns:
        raise ValueError(f"Колонка '{price_col}' не найдена в DataFrame.")

    df_processed = df.copy()
    df_processed['ma'] = df_processed[price_col].rolling(window=ma_window).mean()
    df_processed.dropna(inplace=True) # Удаляем строки с NaN MA (в начале)

    if df_processed.empty:
        raise ValueError("DataFrame стал пустым после добавления MA и удаления NaN. "
                         "Попробуйте увеличить объем данных или уменьшить окно MA_WINDOW.")

    # Масштабируем 'price' и 'ma' вместе
    values_to_scale = df_processed[[price_col, 'ma']].values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values_to_scale)

    original_prices_for_plotting = df_processed[price_col]
    original_ma_for_plotting = df_processed['ma']
    timestamps = df_processed.index

    print(f"Данные подготовлены: {len(scaled_values)} сэмплов после расчета MA и удаления NaN.")
    return scaled_values, scaler, timestamps, original_prices_for_plotting, original_ma_for_plotting

#%%
def split_sequences(scaled_data, n_steps_in, n_steps_out, timestamps):
    """Разделяет данные на X (последовательности) и y (целевые значения)."""
    X, y, y_timestamps = [], [], []
    # scaled_data имеет форму (samples, n_features)
    # y будет первым признаком (цена)
    for i in range(len(scaled_data) - n_steps_in - n_steps_out + 1):
        X.append(scaled_data[i : i + n_steps_in, :]) # Все признаки для входа
        y.append(scaled_data[i + n_steps_in : i + n_steps_in + n_steps_out, 0]) # Только цена для выхода
        y_timestamps.append(timestamps[i + n_steps_in]) # Временная метка первого предсказанного значения

    if not X:
        raise ValueError(f"Недостаточно данных для создания последовательностей с n_steps_in={n_steps_in} и n_steps_out={n_steps_out}."
                         f"Требуется как минимум {n_steps_in + n_steps_out} точек данных, получено {len(scaled_data)}.")

    return np.array(X), np.array(y), np.array(y_timestamps)

#%%
def build_lstm_model(n_steps_in, n_steps_out, n_features):
    """Создает и компилирует LSTM модель."""
    model = Sequential([
        LSTM(75, return_sequences=True, input_shape=(n_steps_in, n_features)),
        Dropout(0.2),
        LSTM(75),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(n_steps_out) # Предсказывает n_steps_out для целевого признака (цены)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.summary() # Опционально: вывести структуру модели
    return model

#%%
def plot_predictions(dates, actual, predicted, title, ma_actual=None, ma_predicted=None):
    """Отображает фактические и предсказанные значения, опционально с MA."""
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Фактическая цена', color='blue', alpha=0.8)
    plt.plot(dates, predicted, label='Предсказанная цена', color='red', linestyle='--', alpha=0.8)
    if ma_actual is not None:
        plt.plot(dates, ma_actual, label='Фактическая MA', color='cyan', linestyle=':', alpha=0.6)
    if ma_predicted is not None:
         plt.plot(dates, ma_predicted, label='MA предсказанных цен', color='orange', linestyle='-.', alpha=0.6)

    plt.title(title)
    plt.xlabel("Дата и Время")
    plt.ylabel("Цена")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#%%
def inverse_transform_predictions(scaled_preds, scaler, n_features):
    """
    Выполняет обратное масштабирование для предсказаний первого признака (цены).
    scaled_preds: форма (num_samples, n_steps_out) или (num_samples, 1) если только первый шаг.
    """
    if scaled_preds.ndim == 1:
        scaled_preds = scaled_preds.reshape(-1, 1) # Гарантируем 2D: (samples, 1 step)

    num_samples, num_steps = scaled_preds.shape[0], scaled_preds.shape[1]

    # Создаем фиктивный массив с тем же количеством признаков, на котором обучался scaler
    # Преобразуем предсказания в форму (num_samples * num_steps, 1)
    preds_flat = scaled_preds.reshape(-1, 1)
    dummy_array = np.zeros((preds_flat.shape[0], n_features), dtype=np.float32)
    dummy_array[:, 0] = preds_flat[:, 0] # Заполняем первую колонку (цена) предсказаниями

    # Обратное масштабирование
    inversed_flat = scaler.inverse_transform(dummy_array)[:, 0]

    # Возвращаем к исходной форме (num_samples, num_steps)
    return inversed_flat.reshape(num_samples, num_steps)

#%%
def cross_validate_lstm(X_trainval, y_trainval, y_times_trainval, scaler, n_features, n_steps_in, n_steps_out, n_splits=4):
    """Выполняет K-блочную кросс-валидацию."""
    best_mse = float('inf')
    best_model = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE) # shuffle=True для разнообразия фолдов

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        print(f"\n--- Кросс-валидация: Фолд {fold+1}/{n_splits} ---")
        model = build_lstm_model(n_steps_in, n_steps_out, n_features)

        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]
        y_times_val = y_times_trainval[val_idx]

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, verbose=0,
                            callbacks=[early_stopping])

        # Оценка на первом шаге предсказания
        preds_scaled = model.predict(X_val) # Форма: (n_samples, N_STEPS_OUT)
        preds_first_step_scaled = preds_scaled[:, 0] # Форма: (n_samples,)
        y_val_first_step_scaled = y_val[:, 0]    # Форма: (n_samples,)

        preds_first_step_inv = inverse_transform_predictions(preds_first_step_scaled, scaler, n_features).flatten()
        y_val_first_step_inv = inverse_transform_predictions(y_val_first_step_scaled, scaler, n_features).flatten()

        mse = mean_squared_error(y_val_first_step_inv, preds_first_step_inv)
        print(f"Фолд {fold+1} MSE (1-й шаг): {mse:.4f}")

        if fold == 0: # График только для первого фолда для экономии времени/места
            plot_predictions(y_times_val, y_val_first_step_inv, preds_first_step_inv,
                             title=f"Фолд {fold+1} Валидация - Предсказание 1-го шага")

        if mse < best_mse:
            best_mse = mse
            best_model = model
            print(f"Найдена новая лучшая модель в фолде {fold+1} с MSE: {best_mse:.4f}")

    print(f"\nЛучшая MSE кросс-валидации (1-й шаг): {best_mse:.4f}")
    return best_model

#%%
def evaluate_on_test_set(model, X_test, y_test, y_times_test, scaler, n_features,
                         original_prices_on_test, original_ma_on_test):
    """Оценивает модель на тестовом наборе и отображает результаты."""
    print("\n--- Оценка на тестовом наборе ---")
    preds_scaled = model.predict(X_test) # (samples, N_STEPS_OUT)

    # Оценка и график для первого шага предсказания
    preds_first_step_scaled = preds_scaled[:, 0]
    y_test_first_step_scaled = y_test[:, 0]

    preds_inv = inverse_transform_predictions(preds_first_step_scaled, scaler, n_features).flatten()
    actual_inv = inverse_transform_predictions(y_test_first_step_scaled, scaler, n_features).flatten()

    mse = mean_squared_error(actual_inv, preds_inv)
    print(f"Тестовая MSE (1-й шаг): {mse:.4f}")

    plot_predictions(y_times_test, actual_inv, preds_inv,
                     title="Тестовый набор - Предсказание 1-го шага vs Факт",
                     ma_actual=original_ma_on_test if original_ma_on_test is not None else None)

    return preds_inv, actual_inv, mse

#%%
def generate_future_forecast(model, last_sequence_scaled, scaler, n_features,
                             n_steps_in, n_steps_out, forecast_horizon_minutes,
                             ma_window_forecast, last_timestamp):
    """Генерирует прогноз на будущие временные шаги без новых данных API."""
    print(f"\n--- Генерация прогноза на {forecast_horizon_minutes} минут вперед ---")

    current_sequence = last_sequence_scaled.reshape(1, n_steps_in, n_features)

    all_future_prices_inv = []
    all_future_times = []

    # Цикл для итеративного прогнозирования, если N_STEPS_OUT < forecast_horizon_minutes
    # В данном упрощенном варианте, мы делаем один прогноз на N_STEPS_OUT
    # и берем из него нужное количество точек.
    # Для полноценного итеративного прогноза логика была бы сложнее.

    future_preds_scaled_steps = model.predict(current_sequence) # Форма (1, N_STEPS_OUT)

    num_predictions_to_take = min(n_steps_out, forecast_horizon_minutes)

    forecast_prices_scaled_segment = future_preds_scaled_steps[0, :num_predictions_to_take]

    forecast_prices_inv_segment = inverse_transform_predictions(forecast_prices_scaled_segment, scaler, n_features).flatten()

    all_future_prices_inv.extend(forecast_prices_inv_segment)

    # Генерация будущих временных меток (интервал 1 минута предполагается CANDLE_TIMEFRAME)
    current_last_time = last_timestamp
    for i in range(num_predictions_to_take):
        current_last_time += timedelta(minutes=1)
        all_future_times.append(current_last_time)

    forecast_prices_inv = np.array(all_future_prices_inv)
    future_times = pd.to_datetime(all_future_times)

    # Расчет MA для этих предсказанных будущих цен
    forecast_ma = pd.Series(forecast_prices_inv).rolling(
        window=min(ma_window_forecast, len(forecast_prices_inv)-1 if len(forecast_prices_inv)>1 else 1),
        min_periods=1).mean().values

    print(f"Сгенерировано {len(forecast_prices_inv)} будущих ценовых точек.")

    plot_predictions(future_times, forecast_prices_inv, forecast_prices_inv, # факт=предсказание для прогноза
                     title=f"Независимый прогноз на {num_predictions_to_take} минут",
                     ma_predicted=forecast_ma)

    return future_times, forecast_prices_inv, forecast_ma

#%%
def main():
    # 1. Загрузка данных
    df_raw = get_tinkoff_candles(TINKOFF_API_TOKEN, INSTRUMENT_FIGI, DAYS_HISTORY, CANDLE_TIMEFRAME)
    if df_raw.empty:
        print("Нет данных для продолжения.")
        return

    # 2. Добавление MA и масштабирование данных
    try:
        scaled_values, scaler, all_original_timestamps, original_prices_series, original_ma_series = \
            add_ma_and_scale_data(df_raw, PRICE_COLUMN, MA_WINDOW)
    except ValueError as e:
        print(f"Ошибка при подготовке данных: {e}")
        return

    # 3. Разделение на последовательности
    try:
        X_all, y_all, y_timestamps_all = split_sequences(scaled_values, N_STEPS_IN, N_STEPS_OUT, all_original_timestamps)
    except ValueError as e:
        print(f"Ошибка при разделении на последовательности: {e}")
        return

    # 4. Разделение на обучающий (+валидационный) и тестовый наборы
    # shuffle=False важно для временных рядов при основном разделении
    X_trainval, X_test, y_trainval, y_test, y_times_trainval, y_times_test = train_test_split(
        X_all, y_all, y_timestamps_all, test_size=0.2, shuffle=False
    )

    # Получение соответствующих оригинальных (немасштабированных) цен и MA для тестового набора
    # y_times_test содержит временные метки, для которых мы делаем предсказания в тестовом наборе
    original_prices_for_test_actuals = original_prices_series.loc[y_times_test].values
    original_ma_for_test_actuals = original_ma_series.loc[y_times_test].values

    print(f"Размер обучающего/валидационного набора: {len(X_trainval)} сэмплов")
    print(f"Размер тестового набора: {len(X_test)} сэмплов")

    # 5. Кросс-валидация и обучение лучшей модели
    best_model = cross_validate_lstm(X_trainval, y_trainval, y_times_trainval, scaler, N_FEATURES,
                                     N_STEPS_IN, N_STEPS_OUT, n_splits=3) # n_splits=3 для ускорения
    if best_model is None:
        print("Обучение модели не удалось.")
        return

    # 6. Оценка на тестовом наборе
    test_preds_inv, test_actual_inv, test_mse = evaluate_on_test_set(
        best_model, X_test, y_test, y_times_test, scaler, N_FEATURES,
        original_prices_for_test_actuals,
        original_ma_for_test_actuals
    )

    # 7. Отображение общей производительности на тестовом наборе
    test_results_df = pd.DataFrame({
        'Actual Close': test_actual_inv,
        'Predicted Close': test_preds_inv
    }, index=y_times_test)
    test_results_df.sort_index(inplace=True)

    plt.figure(figsize=(15, 7))
    plt.plot(test_results_df.index, test_results_df['Actual Close'], label='Фактическая цена (Тест)', color='dodgerblue')
    plt.plot(test_results_df.index, test_results_df['Predicted Close'], label='Предсказанная цена (Тест - 1-й шаг)', color='darkorange', linestyle='--')

    if original_ma_for_test_actuals is not None and len(original_ma_for_test_actuals) == len(test_results_df.index):
        plt.plot(test_results_df.index, original_ma_for_test_actuals, label='Фактическая MA (Тест)', color='limegreen', linestyle=':', alpha=0.7)

    plt.title('Производительность модели на тестовом наборе (1-й шаг предсказания)')
    plt.xlabel('Дата и Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 8. Генерация будущего прогноза (без данных API)
    # Используем последнюю последовательность из scaled_values для инициализации прогноза
    true_last_sequence_for_forecast_scaled = scaled_values[-N_STEPS_IN:, :].reshape(1, N_STEPS_IN, N_FEATURES)
    true_last_data_timestamp = all_original_timestamps[-1]

    generate_future_forecast(best_model, true_last_sequence_for_forecast_scaled, scaler, N_FEATURES,
                               N_STEPS_IN, N_STEPS_OUT, FORECAST_HORIZON_MINUTES,
                               MA_WINDOW, true_last_data_timestamp)

if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True) # Полезно для отладки, но может замедлить выполнение

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Используется GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU не найден, используется CPU.")

    main()
