import os
import random
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- КОНФИГУРАЦИЯ СИМУЛЯЦИИ ---
FIGI_SIM = "BBG004730N88"
OUTPUT_DIRECTORY = "trading_simulation_output_v6_glued"
INITIAL_DEPOSIT_RUB = Decimal("100000.0")
START_PRICE = 315.0
LOT_SIZE = 10
COMMISSION_RATE = Decimal("0.0005")
SHORT_ENABLED = True
SIMULATION_DAYS = 5

# Параметры генерации данных
START_DATE = datetime.now().date() - timedelta(days=7)
MOSCOW_TZ = pytz.timezone('Europe/Moscow')
TRADING_START_HM = (10, 0)
TRADING_END_HM = (18, 45)
VOLATILITY = 0.15
SINE_WAVE_AMPLITUDE = 0.8

# Параметры логики робота
DECISION_INTERVAL_MINUTES = 10
FORCE_CLOSE_POSITION_MINUTES = 120

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Создана директория для результатов: {OUTPUT_DIRECTORY}")

# --- Вспомогательные функции (без изменений) ---

def format_datetime_msk(dt_utc: datetime) -> str:
    if dt_utc is None: return "-"
    if dt_utc.tzinfo is None: dt_utc = pytz.utc.localize(dt_utc)
    return dt_utc.astimezone(MOSCOW_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')

def format_decimal(d, precision='0.00'):
    if isinstance(d, (Decimal, float, int)):
        return d.quantize(Decimal(precision), rounding=ROUND_HALF_UP)
    return d

# --- Функции генерации и симуляции (без изменений) ---

def generate_trading_data(day, start_hm, end_hm, start_price, volatility, sine_amplitude):
    start_time = MOSCOW_TZ.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=start_hm[0], minutes=start_hm[1]))
    end_time = MOSCOW_TZ.localize(datetime.combine(day, datetime.min.time()) + timedelta(hours=end_hm[0], minutes=end_hm[1]))
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    n_points = len(timestamps)
    random_walk = np.random.normal(scale=volatility, size=n_points).cumsum()
    sine_wave = sine_amplitude * np.sin(np.linspace(random.uniform(0, np.pi), random.uniform(3*np.pi, 4*np.pi), n_points))
    real_prices = start_price + random_walk + sine_wave
    ratio_trend = np.concatenate((
        np.linspace(1.005, 1.002, n_points // 3),
        np.linspace(1.001, 0.990, n_points // 3),
        np.linspace(0.990, 1.001, n_points - 2 * (n_points // 3))
    ))
    ratio_noise = np.random.normal(scale=0.001, size=n_points)
    predicted_prices = real_prices * (ratio_trend + ratio_noise)
    df = pd.DataFrame({
        'timestamp_utc': timestamps.tz_convert('UTC'),
        'real_price': real_prices,
        'predicted_price': predicted_prices,
        'forecast_real_ratio': predicted_prices / real_prices
    })
    df['real_price'] = df['real_price'].apply(lambda x: Decimal(f"{x:.4f}"))
    df['predicted_price'] = df['predicted_price'].apply(lambda x: Decimal(f"{x:.4f}"))
    return df

def simulate_trading_day(data, initial_cash, initial_shares, initial_avg_price, trade_number_counter, lot_size, commission_rate, short_enabled):
    cash, shares, avg_entry_price = initial_cash, initial_shares, initial_avg_price
    trade_log = []
    start_day_value = cash + shares * data['real_price'].iloc[0]
    portfolio_history = [(data['timestamp_utc'].iloc[0] - timedelta(seconds=1), start_day_value)]
    position_open_time = None
    for index, row in data.iterrows():
        current_time, real_price = row['timestamp_utc'], row['real_price']
        should_buy, should_sell = False, False
        if index % DECISION_INTERVAL_MINUTES == 0:
            future_index = index + DECISION_INTERVAL_MINUTES
            if future_index < len(data):
                predicted_price_in_10_min = data['predicted_price'].iloc[future_index]
                if predicted_price_in_10_min > real_price: should_buy = True
                elif predicted_price_in_10_min < real_price: should_sell = True
        force_close = False
        if position_open_time and (current_time - position_open_time).total_seconds() / 60 > FORCE_CLOSE_POSITION_MINUTES:
            force_close = True
            print(f"{format_datetime_msk(current_time)}: Принудительное закрытие позиции.")
        if (should_buy and shares <= 0) or (force_close and shares < 0):
            lots_to_trade = abs(shares) / lot_size if shares < 0 else int(cash // (real_price * lot_size))
            if lots_to_trade > 0:
                trade_number_counter += 1
                shares_to_trade = lots_to_trade * lot_size
                trade_cost = shares_to_trade * real_price
                commission = trade_cost * commission_rate
                realized_pnl = (avg_entry_price - real_price) * shares_to_trade - commission if shares < 0 else Decimal(0)
                direction_str = "Покупка (закр. шорт)" if shares < 0 else "Покупка (лонг)"
                cash -= (trade_cost + commission)
                prev_shares, prev_avg_price = shares, avg_entry_price
                shares += shares_to_trade
                if shares > 0: avg_entry_price = ((abs(prev_shares) * prev_avg_price) + (shares_to_trade * real_price)) / shares
                else: avg_entry_price = Decimal(0)
                position_open_time = current_time if shares != 0 else None
                trade_log.append({"trade_number": trade_number_counter, "timestamp": current_time, "figi": FIGI_SIM, "direction_str": direction_str, "lots_executed": int(lots_to_trade), "shares_executed": int(shares_to_trade), "executed_price_per_share": real_price, "initial_cost_or_proceeds_rub": trade_cost, "commission_rub": commission, "realized_pnl_this_trade": realized_pnl, "cash_after_trade": cash, "portfolio_value_after_trade": cash + shares * real_price, "success": True})
        elif (should_sell and shares >= 0 and (short_enabled or shares > 0)) or (force_close and shares > 0):
            if shares > 0: lots_to_trade = shares / lot_size
            elif short_enabled: lots_to_trade = int((cash + shares * real_price) / 2 // (real_price * lot_size))
            else: lots_to_trade = 0
            if lots_to_trade > 0:
                trade_number_counter += 1
                shares_to_trade = lots_to_trade * lot_size
                trade_proceeds = shares_to_trade * real_price
                commission = trade_proceeds * commission_rate
                realized_pnl = (real_price - avg_entry_price) * shares_to_trade - commission if shares > 0 else Decimal(0)
                direction_str = "Продажа (закр. лонг)" if shares > 0 else "Продажа (шорт)"
                cash += (trade_proceeds - commission)
                prev_shares, prev_avg_price = shares, avg_entry_price
                shares -= shares_to_trade
                if shares < 0: avg_entry_price = ((abs(prev_shares) * prev_avg_price) + (shares_to_trade * real_price)) / abs(shares)
                else: avg_entry_price = Decimal(0)
                position_open_time = current_time if shares != 0 else None
                trade_log.append({"trade_number": trade_number_counter, "timestamp": current_time, "figi": FIGI_SIM, "direction_str": direction_str, "lots_executed": int(lots_to_trade), "shares_executed": int(shares_to_trade), "executed_price_per_share": real_price, "initial_cost_or_proceeds_rub": trade_proceeds, "commission_rub": commission, "realized_pnl_this_trade": realized_pnl, "cash_after_trade": cash, "portfolio_value_after_trade": cash + shares * real_price, "success": True})
        portfolio_history.append((current_time, cash + (shares * real_price)))
    return trade_log, portfolio_history, cash, shares, avg_entry_price, trade_number_counter

# --- Табличные отчеты (без изменений) ---
def generate_pnl_table_df(trade_log, figi):
    if not trade_log: return pd.DataFrame()
    df_pnl = pd.DataFrame(trade_log).copy()
    df_pnl["Время (MSK)"] = df_pnl["timestamp"].apply(format_datetime_msk)
    df_pnl = df_pnl.rename(columns={"trade_number": "№", "figi": "FIGI", "direction_str": "Операция", "lots_executed": "Лоты", "shares_executed": "Акции"})
    columns_map = [("Цена за акцию", "executed_price_per_share", '0.0000', 10),("Стоимость/Выручка (без ком.)", "initial_cost_or_proceeds_rub", '0.00', 12),("Комиссия", "commission_rub", '0.00', 8),("P&L сделки", "realized_pnl_this_trade", '0.00', 10),("Кэш после сделки", "cash_after_trade", '0.00', 15),("Портфель после сделки", "portfolio_value_after_trade", '0.00', 18)]
    for target_col, source_col, prec, align in columns_map:
        if source_col in df_pnl.columns:
            df_pnl[target_col] = df_pnl[source_col].apply(lambda x: f"{format_decimal(x, prec):>{align}}" if x != 0 else "-")
    final_columns_order = ["№", "Время (MSK)", "FIGI", "Операция", "Лоты", "Акции", "Цена за акцию", "Стоимость/Выручка (без ком.)", "Комиссия", "P&L сделки", "Кэш после сделки", "Портфель после сделки"]
    return df_pnl[final_columns_order]

def generate_metrics_table(data_df):
    real, pred = data_df['real_price'].astype(float), data_df['predicted_price'].astype(float)
    return pd.DataFrame({"Метрика": ["MSE", "MAE", "R^2 Score"], "Значение": [f"{mean_squared_error(real, pred):.4f}", f"{mean_absolute_error(real, pred):.4f}", f"{r2_score(real, pred):.4f}"]})


# --- ИЗМЕНЕНИЕ: НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ "СКЛЕИВАНИЯ" ---

def get_continuous_timeline_ticks(timestamps_utc_series):
    """
    Создает метки для оси X, чтобы график выглядел непрерывным.
    Возвращает позиции меток и их текстовые подписи.
    """
    tick_locs = []
    tick_labels = []
    last_day = None

    for i, ts in enumerate(timestamps_utc_series):
        ts_msk = ts.astimezone(MOSCOW_TZ)
        current_day = ts_msk.date()
        # Ставим метку в начале каждого нового дня
        if current_day != last_day:
            tick_locs.append(i)
            # Формат метки: '25-Mar 10:00'
            tick_labels.append(ts_msk.strftime('%d-%b\n%H:%M'))
            last_day = current_day

    return tick_locs, tick_labels


# --- ИЗМЕНЕНИЕ: ОБНОВЛЕННЫЕ ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ ---

def plot_weekly_prices_glued(market_price_history_df, trade_log, figi_main, output_dir):
    """Строит склеенный график цен и сделок за неделю."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))

    # 1. Создаем искусственную непрерывную ось X
    x_coords = np.arange(len(market_price_history_df))
    timestamps_utc = market_price_history_df['timestamp_utc']

    # 2. Строим основные графики цен по искусственной оси
    plt.plot(x_coords, market_price_history_df['real_price'], label="Рыночная цена", color="blue", zorder=1, linewidth=1.5)
    plt.plot(x_coords, market_price_history_df['predicted_price'], label="Спрогнозированная цена", color="orange", linestyle='--', alpha=0.7, zorder=1)

    # 3. Наносим сделки, находя их позицию на искусственной оси
    if trade_log:
        # Создаем карту для быстрого поиска индекса по времени
        timestamp_to_index = pd.Series(range(len(timestamps_utc)), index=timestamps_utc)
        buy_times_utc = [t["timestamp"] for t in trade_log if "Покупка" in t["direction_str"]]
        buy_indices = timestamp_to_index.reindex(buy_times_utc, method='nearest').tolist()
        buy_prices = [t["executed_price_per_share"] for t in trade_log if "Покупка" in t["direction_str"]]

        sell_times_utc = [t["timestamp"] for t in trade_log if "Продажа" in t["direction_str"]]
        sell_indices = timestamp_to_index.reindex(sell_times_utc, method='nearest').tolist()
        sell_prices = [t["executed_price_per_share"] for t in trade_log if "Продажа" in t["direction_str"]]

        plt.scatter(buy_indices, buy_prices, label="Покупки", color="green", marker="^", s=80, edgecolors='black', zorder=2)
        plt.scatter(sell_indices, sell_prices, label="Продажи", color="red", marker="v", s=80, edgecolors='black', zorder=2)

    # 4. Устанавливаем кастомные метки на ось X
    tick_locs, tick_labels = get_continuous_timeline_ticks(timestamps_utc)
    plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=0, ha='center')

    plt.title(f"Недельный 'склеенный' график цен и сделок для {figi_main}", fontsize=16)
    plt.xlabel("Время (Начало каждого торгового дня)", fontsize=12)
    plt.ylabel("Цена", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    plt.grid(True, which='major', axis='x', linestyle='-', alpha=0.9) # Вертикальные линии для дней
    plt.tight_layout()
    filename = os.path.join(output_dir, f"weekly_glued_prices_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График цен и сделок сохранен: {filename}")
    plt.show()
    plt.close()


def plot_weekly_profitability_glued(portfolio_history, initial_balance, figi_main, output_dir):
    """Строит склеенный график доходности за неделю."""
    if not portfolio_history: return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))

    timestamps_utc, values_decimal = zip(*portfolio_history)
    timestamps_utc_series = pd.Series(timestamps_utc)

    # 1. Создаем искусственную ось X и вычисляем доходность
    x_coords = np.arange(len(values_decimal))
    profit_percent = [((float(v) / float(initial_balance)) - 1) * 100 for v in values_decimal]

    # 2. Строим график
    plt.plot(x_coords, profit_percent, label="Доходность портфеля", color="purple")
    plt.axhline(0, color='grey', linestyle='--', lw=1)

    # 3. Устанавливаем кастомные метки
    tick_locs, tick_labels = get_continuous_timeline_ticks(timestamps_utc_series)
    plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=0, ha='center')

    plt.title(f"Недельный 'склеенный' график доходности для {figi_main}", fontsize=16)
    plt.xlabel("Время (Начало каждого торгового дня)", fontsize=12)
    plt.ylabel("Доходность, %", fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    plt.legend(fontsize=10)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    plt.grid(True, which='major', axis='x', linestyle='-', alpha=0.9)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"weekly_glued_profitability_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График доходности сохранен: {filename}")
    plt.show()
    plt.close()


# --- Основная функция main ---

def main():
    print(f"--- Запуск недельной симуляции ({SIMULATION_DAYS} торговых дней) ---")

    all_days_data_df = pd.DataFrame()
    all_trade_logs = []
    all_portfolio_history = []

    current_cash = INITIAL_DEPOSIT_RUB
    current_shares = Decimal(0)
    current_avg_price = Decimal(0)
    last_close_price = Decimal(START_PRICE)
    trade_counter = 0

    current_day = START_DATE
    days_simulated = 0
    while days_simulated < SIMULATION_DAYS:
        if current_day.weekday() >= 5:
            current_day += timedelta(days=1)
            continue

        print(f"\n--- Симуляция дня {days_simulated + 1}/{SIMULATION_DAYS}: {current_day.strftime('%Y-%m-%d')} ---")

        daily_data_df = generate_trading_data(
            day=current_day, start_hm=TRADING_START_HM, end_hm=TRADING_END_HM,
            start_price=float(last_close_price), volatility=VOLATILITY, sine_amplitude=SINE_WAVE_AMPLITUDE
        )
        print(f"Сгенерировано {len(daily_data_df)} точек. Начальная цена: {daily_data_df['real_price'].iloc[0]:.2f}")

        (daily_trade_log, daily_portfolio_history,
         end_of_day_cash, end_of_day_shares, end_of_day_avg_price,
         trade_counter) = simulate_trading_day(
            data=daily_data_df, initial_cash=current_cash, initial_shares=current_shares,
            initial_avg_price=current_avg_price, trade_number_counter=trade_counter,
            lot_size=LOT_SIZE, commission_rate=COMMISSION_RATE, short_enabled=SHORT_ENABLED
        )

        all_days_data_df = pd.concat([all_days_data_df, daily_data_df], ignore_index=True)
        all_trade_logs.extend(daily_trade_log)
        if days_simulated > 0:
            all_portfolio_history.extend(daily_portfolio_history[1:])
        else:
            all_portfolio_history.extend(daily_portfolio_history)

        current_cash = end_of_day_cash
        current_shares = end_of_day_shares
        current_avg_price = end_of_day_avg_price
        last_close_price = daily_data_df['real_price'].iloc[-1]
        print(f"Конец дня: Кэш={current_cash:.2f}, Акции={current_shares}, Средняя цена={current_avg_price:.4f}")

        days_simulated += 1
        current_day += timedelta(days=1)

    print("\n--- Недельная симуляция завершена. Генерация отчетов. ---")

    final_portfolio_value = current_cash + current_shares * last_close_price
    total_profit_rub = final_portfolio_value - INITIAL_DEPOSIT_RUB
    total_profit_percent = (total_profit_rub / INITIAL_DEPOSIT_RUB) * 100

    print("\n--- Итоговые результаты за неделю ---")
    print(f"Начальный депозит: {INITIAL_DEPOSIT_RUB:,.2f} RUB")
    print(f"Финальная стоимость портфеля: {final_portfolio_value:,.2f} RUB")
    print(f"Общая прибыль/убыток: {total_profit_rub:,.2f} RUB ({total_profit_percent:.2f}%)")
    print(f"Всего сделок за неделю: {len(all_trade_logs)}")

    pnl_df = generate_pnl_table_df(all_trade_logs, FIGI_SIM)
    if not pnl_df.empty:
        pd.set_option('display.width', 2000); pd.set_option('display.max_colwidth', None); pd.set_option('display.max_rows', 200)
        print("\n--- Таблица сделок и P&L за неделю (Время MSK) ---")
        print(pnl_df.to_string(index=False))
        pnl_filename = os.path.join(OUTPUT_DIRECTORY, f"weekly_pnl_table_{FIGI_SIM}.csv")
        pnl_df.to_csv(pnl_filename, index=False, encoding='utf-8-sig')
        print(f"\nТаблица P&L за неделю сохранена: {pnl_filename}")

    metrics_df = generate_metrics_table(all_days_data_df)
    if not metrics_df.empty:
        print("\n--- Таблица метрик качества прогноза за неделю ---")
        print(metrics_df.to_string(index=False))
        metrics_filename = os.path.join(OUTPUT_DIRECTORY, f"weekly_metrics_table_{FIGI_SIM}.csv")
        metrics_df.to_csv(metrics_filename, index=False, encoding='utf-8-sig')
        print(f"Таблица метрик за неделю сохранена: {metrics_filename}")

    # --- ИЗМЕНЕНИЕ: ВЫЗОВ НОВЫХ ФУНКЦИЙ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ ---
    plot_weekly_prices_glued(all_days_data_df, all_trade_logs, FIGI_SIM, OUTPUT_DIRECTORY)
    plot_weekly_profitability_glued(all_portfolio_history, INITIAL_DEPOSIT_RUB, FIGI_SIM, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
