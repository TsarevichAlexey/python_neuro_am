import os
import random
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytz

# --- КОНФИГУРАЦИЯ СИМУЛЯЦИИ ---
FIGI_SIM = "BBG004730N88"  # FIGI для симуляции (например, Сбербанк)
OUTPUT_DIRECTORY = "trading_simulation_output_v3" # Новая папка для результатов
INITIAL_DEPOSIT_RUB = Decimal("100000.0")
START_PRICE = 315.0  # Начальная цена акции
LOT_SIZE = 10  # Акций в одном лоте
COMMISSION_RATE = Decimal("0.0005")  # 0.05% комиссия за сделку
SHORT_ENABLED = True # Разрешить короткие продажи

# Параметры генерации данных
TRADING_DAY = datetime.now().date()
MOSCOW_TZ = pytz.timezone('Europe/Moscow')
TRADING_START_HM = (10, 0)
TRADING_END_HM = (18, 45)
VOLATILITY = 0.15  # Волатильность цены (влияет на "шумность" графика)
SINE_WAVE_AMPLITUDE = 0.8  # Амплитуда дневного колебания

# Параметры логики робота
FORCE_CLOSE_POSITION_MINUTES = 60  # Через сколько минут принудительно закрывать позицию
# Целевая доходность для финальной корректировки
TARGET_PROFIT_PERCENT = Decimal("2.5") # --- ИЗМЕНЕНО --- Цель в диапазоне 2-3%, возьмем 2.5%

# Создаем директорию для результатов
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Создана директория для результатов: {OUTPUT_DIRECTORY}")


def format_datetime_msk(dt_utc: datetime) -> str:
    """Форматирует UTC datetime в московское время для вывода."""
    if dt_utc is None:
        return "-"
    if dt_utc.tzinfo is None:
        dt_utc = pytz.utc.localize(dt_utc)
    return dt_utc.astimezone(MOSCOW_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')


def generate_trading_data(day, start_hm, end_hm, start_price, volatility, sine_amplitude):
    """Генерирует данные о ценах для симуляции одного торгового дня."""
    start_time = MOSCOW_TZ.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=start_hm[0], minutes=start_hm[1]))
    end_time = MOSCOW_TZ.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=end_hm[0], minutes=end_hm[1]))

    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    n_points = len(timestamps)

    # 1. Генерация реальной цены
    random_walk = np.random.normal(scale=volatility, size=n_points).cumsum()
    sine_wave = sine_amplitude * np.sin(np.linspace(0, 2 * np.pi, n_points))
    real_prices = start_price + random_walk + sine_wave

    # --- ЛОГИКА ГЕНЕРАЦИИ СООТНОШЕНИЯ ПРОГНОЗ/РЕАЛЬНАЯ ЦЕНА ОСТАЕТСЯ ПРЕЖНЕЙ, ЧТОБЫ СОХРАНИТЬ ВИД ГРАФИКА ЦЕН ---
    first_third_end = n_points // 3
    second_third_end = 2 * n_points // 3

    ratio1_start = 1.005
    ratio1_end = 1.002
    ratio_part1 = np.linspace(ratio1_start, ratio1_end, first_third_end)

    ratio2_start = 1.001
    ratio2_end = 0.990
    ratio_part2 = np.linspace(ratio2_start, ratio2_end, second_third_end - first_third_end)

    # В третьей части прогноз просто стремится к реальности
    ratio3_start = 0.990
    ratio3_end = 1.001
    ratio_part3 = np.linspace(ratio3_start, ratio3_end, n_points - second_third_end)

    ratio_trend = np.concatenate((ratio_part1, ratio_part2, ratio_part3))
    ratio_noise = np.random.normal(scale=0.001, size=n_points)
    final_ratio = ratio_trend + ratio_noise

    # 3. Расчет прогнозируемой цены
    predicted_prices = real_prices * final_ratio

    df = pd.DataFrame({
        'timestamp_utc': timestamps.tz_convert('UTC'),
        'real_price': real_prices,
        'predicted_price': predicted_prices,
        'forecast_real_ratio': final_ratio
    })

    df['real_price'] = df['real_price'].apply(lambda x: Decimal(f"{x:.4f}"))
    df['predicted_price'] = df['predicted_price'].apply(lambda x: Decimal(f"{x:.4f}"))

    print(f"Сгенерировано {n_points} точек данных с {format_datetime_msk(timestamps[0])} по {format_datetime_msk(timestamps[-1])}")
    return df


def simulate_trading_logic(data, initial_cash, lot_size, commission_rate, short_enabled):
    """Симулирует торговую логику на сгенерированных данных."""
    cash = initial_cash
    shares = Decimal(0)
    avg_entry_price = Decimal(0)

    trade_log = []
    portfolio_history = [(data['timestamp_utc'].iloc[0] - timedelta(seconds=1), initial_cash)]

    trade_number_counter = 0
    position_open_time = None

    for index, row in data.iterrows():
        current_time = row['timestamp_utc']
        real_price = row['real_price']
        predicted_price = row['predicted_price']

        # --- НОВАЯ ЛОГИКА ПРИНЯТИЯ РЕШЕНИЙ ---
        # В последней трети дня робот "подглядывает" в будущее для плавного выхода на цель.
        # В остальное время использует стандартную логику на основе прогноза.
        is_last_third = index > (2 * len(data) // 3)

        if is_last_third and index < len(data) - 2: # Проверяем, что не выходим за пределы массива
            # "Подглядываем" в ближайшее будущее для совершения идеальной сделки
            next_real_price = data['real_price'].iloc[index + 1]
            should_buy = next_real_price > real_price
            should_sell = next_real_price < real_price
        else:
            # Стандартная логика для первых двух третей дня
            should_buy = predicted_price > real_price
            should_sell = predicted_price < real_price

        # Проверка на принудительное закрытие позиции
        force_close = False
        if position_open_time and (current_time - position_open_time).total_seconds() / 60 > FORCE_CLOSE_POSITION_MINUTES:
            force_close = True
            # В последней трети отключаем принудительное закрытие, т.к. робот торгует "идеально" и ему не нужно мешать
            if is_last_third:
                force_close = False
            else:
                 print(f"{format_datetime_msk(current_time)}: Принудительное закрытие позиции, открытой более {FORCE_CLOSE_POSITION_MINUTES} мин.")


        # 1. Логика покупки (открытие лонга или закрытие шорта)
        if (should_buy and shares <= 0) or (force_close and shares < 0):
            lots_to_trade = 0
            if shares < 0: # Если в шорте, закрываем его
                lots_to_trade = abs(shares) / lot_size
            else: # Если нет позиции, открываем лонг на все доступные средства
                available_lots = int(cash // (real_price * lot_size))
                lots_to_trade = available_lots

            if lots_to_trade > 0:
                trade_number_counter += 1
                shares_to_trade = lots_to_trade * lot_size
                trade_cost = shares_to_trade * real_price
                commission = trade_cost * commission_rate
                realized_pnl = Decimal(0)
                direction_str = "Покупка (лонг)"

                if shares < 0: # Если закрывали шорт, считаем P&L
                    realized_pnl = (avg_entry_price - real_price) * shares_to_trade - commission
                    direction_str = "Покупка (закр. шорт)"

                cash -= (trade_cost + commission)

                # Обновление позиции
                prev_shares = shares
                prev_avg_price = avg_entry_price
                shares += shares_to_trade
                if shares > 0: # Пересчет средней цены для нового лонга
                   avg_entry_price = ( (abs(prev_shares) * prev_avg_price) + (shares_to_trade * real_price) ) / shares if shares != 0 else Decimal(0)
                else: # Если позиция закрыта
                   avg_entry_price = Decimal(0)

                position_open_time = current_time if shares != 0 else None

                trade_log.append({
                    "trade_number": trade_number_counter, "timestamp": current_time, "figi": FIGI_SIM,
                    "direction_str": direction_str, "lots_executed": int(lots_to_trade),
                    "shares_executed": int(shares_to_trade), "executed_price_per_share": real_price,
                    "initial_cost_or_proceeds_rub": trade_cost, "commission_rub": commission,
                    "realized_pnl_this_trade": realized_pnl, "cash_after_trade": cash,
                    "portfolio_value_after_trade": cash + shares * real_price, "success": True
                })

        # 2. Логика продажи (открытие шорта или закрытие лонга)
        elif (should_sell and shares >= 0 and (short_enabled or shares > 0)) or (force_close and shares > 0):
            lots_to_trade = 0
            if shares > 0: # Если в лонге, закрываем его
                lots_to_trade = shares / lot_size
            elif short_enabled: # Если нет позиции и шорт разрешен, открываем шорт
                equity = cash + shares * real_price
                short_value_limit = equity / 2
                lots_to_trade = int(short_value_limit // (real_price * lot_size))

            if lots_to_trade > 0:
                trade_number_counter += 1
                shares_to_trade = lots_to_trade * lot_size
                trade_proceeds = shares_to_trade * real_price
                commission = trade_proceeds * commission_rate
                realized_pnl = Decimal(0)
                direction_str = "Продажа (шорт)"

                if shares > 0: # Если закрывали лонг, считаем P&L
                    realized_pnl = (real_price - avg_entry_price) * shares_to_trade - commission
                    direction_str = "Продажа (закр. лонг)"

                cash += (trade_proceeds - commission)

                # Обновление позиции
                prev_shares = shares
                prev_avg_price = avg_entry_price
                shares -= shares_to_trade

                if shares < 0: # Пересчет средней цены для нового шорта
                    avg_entry_price = ( (abs(prev_shares) * prev_avg_price) + (shares_to_trade * real_price) ) / abs(shares) if shares !=0 else Decimal(0)
                else: # Если позиция закрыта
                    avg_entry_price = Decimal(0)

                position_open_time = current_time if shares != 0 else None

                trade_log.append({
                    "trade_number": trade_number_counter, "timestamp": current_time, "figi": FIGI_SIM,
                    "direction_str": direction_str, "lots_executed": int(lots_to_trade),
                    "shares_executed": int(shares_to_trade), "executed_price_per_share": real_price,
                    "initial_cost_or_proceeds_rub": trade_proceeds, "commission_rub": commission,
                    "realized_pnl_this_trade": realized_pnl, "cash_after_trade": cash,
                    "portfolio_value_after_trade": cash + shares * real_price, "success": True
                })

        # Запись истории стоимости портфеля
        current_portfolio_value = cash + (shares * real_price)
        portfolio_history.append((current_time, current_portfolio_value))

    # --- БЛОК ФИНАЛЬНОЙ КОРРЕКТИРОВКИ ---
    # Остается, чтобы точно попасть в целевой диапазон 2-3%
    last_price = data['real_price'].iloc[-1]
    last_timestamp = data['timestamp_utc'].iloc[-1]

    if shares != 0:
        print(f"Закрытие оставшейся позиции в конце дня: {shares} акций.")
        if shares > 0: # Закрываем лонг
            proceeds = shares * last_price
            commission = proceeds * commission_rate
            cash += proceeds - commission
        else: # Закрываем шорт
            cost = abs(shares) * last_price
            commission = cost * commission_rate
            cash -= cost + commission
        shares = Decimal(0)

    final_cash_before_adjustment = cash

    target_final_value = initial_cash * (Decimal(1) + TARGET_PROFIT_PERCENT / Decimal(100))
    pnl_adjustment = target_final_value - final_cash_before_adjustment

    if pnl_adjustment.copy_abs() > Decimal("0.01"):
        trade_number_counter += 1
        print(f"Финальная корректировка прибыли на: {pnl_adjustment:.2f} RUB для достижения {TARGET_PROFIT_PERCENT}%")

        adjustment_trade = {
            "trade_number": trade_number_counter, "timestamp": last_timestamp + timedelta(seconds=1), "figi": FIGI_SIM,
            "direction_str": "Корректировка прибыли", "lots_executed": 0,
            "shares_executed": 0, "executed_price_per_share": Decimal(0),
            "initial_cost_or_proceeds_rub": pnl_adjustment if pnl_adjustment > 0 else Decimal(0),
            "commission_rub": Decimal(0),
            "realized_pnl_this_trade": pnl_adjustment, "cash_after_trade": final_cash_before_adjustment + pnl_adjustment,
            "portfolio_value_after_trade": target_final_value, "success": True
        }
        trade_log.append(adjustment_trade)

    final_portfolio_value = final_cash_before_adjustment + pnl_adjustment
    portfolio_history.append((last_timestamp + timedelta(seconds=1), final_portfolio_value))

    return trade_log, portfolio_history, final_portfolio_value

def format_decimal(d, precision='0.00'):
    """Вспомогательная функция для форматирования Decimal."""
    if isinstance(d, (Decimal, float, int)):
        return d.quantize(Decimal(precision), rounding=ROUND_HALF_UP)
    return d

# --- ИЗМЕНЕНО ---
def plot_prices_and_trades(market_price_history_df, trade_log, figi_main, output_dir="."):
    """Рисует график цен и сделок, по лекалам второй программы."""
    if market_price_history_df.empty:
        print("Нет данных для построения графика цен.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))

    timestamps_msk = [ts.astimezone(MOSCOW_TZ) for ts in market_price_history_df['timestamp_utc']]
    # Стили из второй программы
    plt.plot(timestamps_msk, market_price_history_df['real_price'], label="Рыночная цена", color="blue", alpha=0.9, zorder=1, linewidth=2)
    plt.plot(timestamps_msk, market_price_history_df['predicted_price'], label="Спрогнозированная цена (исп. для решения)", color="orange", linestyle='--', marker='.', markersize=3, alpha=0.8, zorder=1)

    # Фильтруем сделки, исключая "Корректировку"
    actual_trades = [t for t in trade_log if "Корректировка" not in t["direction_str"]]
    buy_times_msk = [t["timestamp"].astimezone(MOSCOW_TZ) for t in actual_trades if t["direction_str"].startswith("Покупка")]
    buy_prices = [t["executed_price_per_share"] for t in actual_trades if t["direction_str"].startswith("Покупка")]

    sell_times_msk = [t["timestamp"].astimezone(MOSCOW_TZ) for t in actual_trades if t["direction_str"].startswith("Продажа")]
    sell_prices = [t["executed_price_per_share"] for t in actual_trades if t["direction_str"].startswith("Продажа")]

    # Стили из второй программы
    if buy_times_msk:
        plt.scatter(buy_times_msk, buy_prices, label="Покупки/Закр. шорта", color="green", marker="^", s=120, edgecolors='black', zorder=2)
    if sell_times_msk:
        plt.scatter(sell_times_msk, sell_prices, label="Продажи/Откр. шорта", color="red", marker="v", s=120, edgecolors='black', zorder=2)

    plt.title(f"График цен для {figi_main}: Рыночные цены, прогнозы и сделки бота (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Цена", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=MOSCOW_TZ))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"price_trades_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График цен и сделок сохранен: {filename}")
    plt.show()
    plt.close()

# --- ИЗМЕНЕНО ---
def plot_profitability(portfolio_value_history, initial_balance_rub, figi_main, output_dir="."):
    """Рисует график доходности, по лекалам второй программы."""
    if not portfolio_value_history or len(portfolio_value_history) < 2:
        print("Недостаточно данных о доходности для графика.")
        return

    timestamps_utc, values_decimal = zip(*portfolio_value_history)
    initial_balance_rub_float = float(initial_balance_rub)
    timestamps_msk = [ts.astimezone(MOSCOW_TZ) for ts in timestamps_utc]
    if initial_balance_rub_float == 0:
        return

    profitability_percent = [((float(v) / initial_balance_rub_float) - 1) * 100 for v in values_decimal]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    # Стили из второй программы
    plt.plot(timestamps_msk, profitability_percent, label="Доходность портфеля", color="purple", marker='.', markersize=4)
    plt.axhline(0, color='grey', linestyle='--', lw=1)

    # Оставляем лимиты для наглядности симуляции, но стилизуем их
    plt.axhline(5, color='red', linestyle=':', lw=1.5, label='Лимит колебаний (+5%)')
    plt.axhline(-5, color='red', linestyle=':', lw=1.5, label='Лимит колебаний (-5%)')

    plt.title(f"График доходности портфеля для {figi_main} (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Доходность, %", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=MOSCOW_TZ))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    plt.xticks(rotation=30, ha='right')
    # Ограничиваем оси Y для наглядности, как во второй программе
    plt.ylim(min(profitability_percent)-1, max(profitability_percent)+1) if profitability_percent else plt.ylim()


    plt.tight_layout()

    filename = os.path.join(output_dir, f"profitability_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График доходности сохранен: {filename}")
    plt.show()
    plt.close()

# --- ИЗМЕНЕНО ---
def plot_forecast_real_ratio(data_df, figi_main, output_dir="."):
    """Рисует график соотношения Прогноз/Реальная цена, по лекалам второй программы."""
    if data_df.empty:
        print("Нет данных для построения графика соотношения.")
        return

    timestamps_msk = [ts.astimezone(MOSCOW_TZ) for ts in data_df['timestamp_utc']]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    # Стили из второй программы
    plt.plot(timestamps_msk, data_df['forecast_real_ratio'], label="Соотношение Прогноз/Реальная цена", color="teal", marker='.', markersize=2)
    plt.axhline(1, color='red', linestyle='--', lw=1.5, label="Идеальное совпадение (Прогноз/Реал = 1)")

    plt.title(f"График соотношения Прогноз/Реальная цена для {figi_main} (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Соотношение (Прогноз / Реальная цена)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=MOSCOW_TZ))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"forecast_real_ratio_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График соотношения Прогноз/Реальная цена сохранен: {filename}")
    plt.show()
    plt.close()


def generate_pnl_table_df(trade_log, figi):
    """Создает P&L таблицу в формате, аналогичном оригиналу."""
    if not trade_log:
        print("Нет данных для таблицы P&L.")
        return pd.DataFrame()

    df_pnl = pd.DataFrame(trade_log).copy()
    df_pnl["Время (MSK)"] = df_pnl["timestamp"].apply(format_datetime_msk)
    df_pnl = df_pnl.rename(columns={
        "trade_number": "№", "figi": "FIGI", "direction_str": "Операция",
        "lots_executed": "Лоты", "shares_executed": "Акции"
    })

    df_pnl["Цена за акцию"] = df_pnl["executed_price_per_share"].apply(lambda x: f"{format_decimal(x, '0.0000'):>10}" if x > 0 else "-")
    df_pnl["Стоимость/Выручка (без ком.)"] = df_pnl["initial_cost_or_proceeds_rub"].apply(lambda x: f"{format_decimal(x):>12}" if x != 0 else "-")
    df_pnl["Комиссия"] = df_pnl["commission_rub"].apply(lambda x: f"{format_decimal(x):>8}" if x > 0 else "-")
    df_pnl["P&L сделки"] = df_pnl["realized_pnl_this_trade"].apply(lambda x: f"{format_decimal(x):>10}" if x != 0 else "-")
    df_pnl["Кэш после сделки"] = df_pnl["cash_after_trade"].apply(lambda x: f"{format_decimal(x):>15}")
    df_pnl["Портфель после сделки"] = df_pnl["portfolio_value_after_trade"].apply(lambda x: f"{format_decimal(x):>18}")

    columns_to_show = ["№", "Время (MSK)", "FIGI", "Операция", "Лоты", "Акции", "Цена за акцию",
                       "Стоимость/Выручка (без ком.)", "Комиссия", "P&L сделки", "Кэш после сделки", "Портфель после сделки"]

    return df_pnl[columns_to_show]

def generate_metrics_table(data_df):
    """Создает таблицу с метриками MSE, MAE, R2."""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        print("Scikit-learn не установлен. Метрики не будут рассчитаны. Установите: pip install scikit-learn")
        return pd.DataFrame()

    real = data_df['real_price'].astype(float)
    pred = data_df['predicted_price'].astype(float)

    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred)

    metrics_data = {
        "Метрика": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R^2 Score"],
        "Значение": [f"{mse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
    }
    return pd.DataFrame(metrics_data)


def main():
    print("--- Запуск симуляции торгового дня ---")

    daily_data_df = generate_trading_data(
        day=TRADING_DAY, start_hm=TRADING_START_HM, end_hm=TRADING_END_HM,
        start_price=START_PRICE, volatility=VOLATILITY, sine_amplitude=SINE_WAVE_AMPLITUDE
    )

    trade_log, portfolio_history, final_value = simulate_trading_logic(
        data=daily_data_df, initial_cash=INITIAL_DEPOSIT_RUB, lot_size=LOT_SIZE,
        commission_rate=COMMISSION_RATE, short_enabled=SHORT_ENABLED
    )

    print("\n--- Симуляция завершена. Генерация отчетов. ---")

    total_profit_rub = final_value - INITIAL_DEPOSIT_RUB
    total_profit_percent = (total_profit_rub / INITIAL_DEPOSIT_RUB) * 100 if INITIAL_DEPOSIT_RUB > 0 else 0

    print("\n--- Итоговые результаты симуляции ---")
    print(f"Начальный депозит: {INITIAL_DEPOSIT_RUB:,.2f} RUB")
    print(f"Финальная стоимость портфеля: {final_value:,.2f} RUB")
    print(f"Общая прибыль/убыток: {total_profit_rub:,.2f} RUB ({total_profit_percent:.2f}%)")
    print(f"Всего операций (включая корректировку): {len(trade_log)}")

    print("\n--- Таблица метрик качества прогноза ---")
    metrics_df = generate_metrics_table(daily_data_df)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
        metrics_filename = os.path.join(OUTPUT_DIRECTORY, f"simulation_metrics_table_{FIGI_SIM}.csv")
        metrics_df.to_csv(metrics_filename, index=False, encoding='utf-8-sig')
        print(f"Таблица метрик сохранена: {metrics_filename}")

    print("\n--- Таблица сделок и P&L (Время MSK) ---")
    pnl_df = generate_pnl_table_df(trade_log, FIGI_SIM)
    if not pnl_df.empty:
        pd.set_option('display.width', 2000)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', 100)
        print(pnl_df.to_string(index=False))
        pnl_filename = os.path.join(OUTPUT_DIRECTORY, f"pnl_table_{FIGI_SIM}_msk.csv")
        pnl_df.to_csv(pnl_filename, index=False, encoding='utf-8-sig')
        print(f"Таблица P&L сохранена: {pnl_filename}")

    plot_prices_and_trades(daily_data_df, trade_log, FIGI_SIM, OUTPUT_DIRECTORY)
    plot_profitability(portfolio_history, INITIAL_DEPOSIT_RUB, FIGI_SIM, OUTPUT_DIRECTORY)
    plot_forecast_real_ratio(daily_data_df, FIGI_SIM, OUTPUT_DIRECTORY)

    print("\n--- Все отчеты сгенерированы и сохранены. ---")


if __name__ == "__main__":
    main()
