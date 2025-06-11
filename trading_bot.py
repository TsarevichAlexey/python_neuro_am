#%%
from datetime import datetime, timedelta, timezone
import time, os
import pandas as pd
from glob import glob
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero
import pytz

# Попытка импорта inputimeout
try:
    from inputimeout import inputimeout, TimeoutOccurred

    INPUTIMEOUT_AVAILABLE = True
except ImportError:
    INPUTIMEOUT_AVAILABLE = False
    print("ПРЕДУПРЕЖДЕНИЕ: Библиотека 'inputimeout' не найдена. Интерактивный запрос на продолжение не будет работать.")
    print("Робот будет продолжать работу автоматически каждые 10 минут без запроса.")
    print("Для интерактивного запроса установите 'inputimeout': pip install inputimeout")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tinkoff.invest import (
    Client,
    OrderDirection,
    OrderType,
    RequestError,
    Quotation,
    MoneyValue,
    InstrumentIdType,
    OrderExecutionReportStatus
)
from tinkoff.invest.utils import (
    now,
    quotation_to_decimal,
    decimal_to_quotation,
    money_to_decimal
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- НОВЫЕ ИМПОРТЫ ДЛЯ МЕТРИК ---
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ПРЕДУПРЕЖДЕНИЕ: Библиотека 'scikit-learn' не найдена. Расчет метрик MSE, MAE, R2 не будет работать.")
    print("Вы можете установить её командой: pip install scikit-learn")

# --- ИМПОРТ ДЛЯ JUPYTER DISPLAY ---
try:
    from IPython.display import display

    JUPYTER_DISPLAY_AVAILABLE = True
except ImportError:
    JUPYTER_DISPLAY_AVAILABLE = False
    print("ПРЕДУПРЕЖДЕНИЕ: IPython.display не доступен. Вывод датафреймов в Jupyter будет через print().")


    # Создаем заглушку для display, если IPython нет
    def display(obj, *args, **kwargs):
        print(obj)

INSTRUMENT_FIGI = "BBG004730N88"
FEATURE_COLUMN = 'close'
OUTPUT_CSV_DIRECTORY = "future_price_predictions"

try:
    from config import TINKOFF_SANDBOX_TOKEN

    print(f"DEBUG: Загруженный TINKOFF_SANDBOX_TOKEN: '{TINKOFF_SANDBOX_TOKEN[:5]}...{TINKOFF_SANDBOX_TOKEN[-5:]}'")
    if not TINKOFF_SANDBOX_TOKEN or TINKOFF_SANDBOX_TOKEN == "your_sandbox_token" or not TINKOFF_SANDBOX_TOKEN.startswith(
            "t."):
        print("КРИТИЧЕСКАЯ ОШИБКА: TINKOFF_SANDBOX_TOKEN не валиден или не является токеном песочницы.")
        exit()
except ImportError:
    print("Ошибка: Не удалось импортировать TINKOFF_SANDBOX_TOKEN из 'config.py'.")
    exit()
except AttributeError:
    print("КРИТИЧЕСКАЯ ОШИБКА: TINKOFF_SANDBOX_TOKEN в config.py имеет неверный тип или значение None.")
    exit()

TRADE_INTERVAL_SECONDS = 10 * 60  # 10 минут
USER_PROMPT_TIMEOUT_SECONDS = 90  # 90 секунд на ответ пользователя
PREDICTION_LOOKAHEAD_MINUTES = 10
INITIAL_DEPOSIT_RUB = Decimal("15000.0")
NUM_FILES_TO_LOAD_PER_MODEL_TYPE = 3
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

TARGET_NO_POSITION_SHARES = Decimal(0)

MODEL_NAME_KEY_PARTS = [
    "Model_1min_TF_30d_hist_30m_",
    "Model_5min_TF_60d_hist_10h_",
    "Model_10min_TF_90d_hist_1day_"
]
print(f"INFO: Робот будет искать прогнозы для следующих ключевых частей имен моделей: {MODEL_NAME_KEY_PARTS}")

PRICE_STAGNATION_TIMEOUT_MINUTES = 20

CONTINUOUS_DEVIATION_THRESHOLD_PERCENT = Decimal("5.0")
CONTINUOUS_DEVIATION_DURATION_MINUTES = 20
INSTANT_DEVIATION_THRESHOLD_PERCENT = Decimal("10.0")

# --- НОВАЯ КОНСТАНТА ДЛЯ ПОРОГА R2 ---
# Если R2 падает ниже этого значения, робот НЕ остановится, а выведет предупреждение.
R2_ACCURACY_THRESHOLD_STOP = Decimal("0.1")
MIN_SAMPLES_FOR_METRICS = 5  # Минимальное количество наблюдений для расчета метрик

if not os.path.exists(OUTPUT_CSV_DIRECTORY):
    os.makedirs(OUTPUT_CSV_DIRECTORY, exist_ok=True)
    print(f"Создана директория для прогнозов и отчетов: {OUTPUT_CSV_DIRECTORY}")


def format_datetime_msk(dt_utc: datetime) -> str:
    if dt_utc is None:
        return "-"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(MOSCOW_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')


def create_money_value(value: Decimal, currency_code: str) -> MoneyValue:
    value = value.quantize(Decimal('0.000000001'), rounding=ROUND_HALF_UP)
    units = int(value)
    nano = int((value - Decimal(units)) * Decimal("1000000000"))
    return MoneyValue(units=units, nano=nano, currency=currency_code.lower())


def find_latest_prediction_files(model_key_part, directory, figi, num_files):
    pattern = os.path.join(directory, f"predictions_{figi}_*{model_key_part}*_*.csv")
    files = glob(pattern)
    if not files: return []
    files.sort(key=lambda x: os.path.basename(x).split('_')[-1].replace('.csv', ''), reverse=True)
    return files[:num_files]


def load_predictions(model_key_parts_list, directory, figi, target_column_name, num_files_per_type):
    all_predictions_data = {}
    print("\n--- Загрузка файлов с прогнозами по типам моделей ---")
    for key_part in model_key_parts_list:
        latest_files = find_latest_prediction_files(key_part, directory, figi, num_files=num_files_per_type)
        if not latest_files:
            print(f"  Файлы прогнозов для типа '{key_part}' не найдены.")
            continue
        model_type_dfs_to_concat = []
        loaded_filenames_for_type = []
        print(f"  Для типа модели (ключ: '{key_part}'):")
        for file_path in latest_files:
            try:
                df = pd.read_csv(file_path)
                # <ВАЛИДАЦИЯ 1.1: После чтения CSV>
                print(f"    --- Проверка файла: {os.path.basename(file_path)} ---")
                print(f"    Колонки до обработки: {list(df.columns)}")
                if JUPYTER_DISPLAY_AVAILABLE:
                    display(df.head(3))
                else:
                    print(df.head(3))
                # print("    Информация о DataFrame (до обработки):")
                # df.info(buf=sys.stdout) # Вывод в stdout для Jupyter
                # </ВАЛИДАЦИЯ 1.1>

                time_col_found = False
                possible_time_cols = ['future_timestamp_utc', 'future_timestamp', 'timestamp_utc', 'timestamp']
                actual_time_col = None
                for col_name in possible_time_cols:
                    if col_name in df.columns:
                        actual_time_col = col_name
                        time_col_found = True
                        break

                if not time_col_found:
                    print(
                        f"    Предупреждение: Ни одна из стандартных колонок времени ({', '.join(possible_time_cols)}) не найдена в {file_path}. Пропуск файла.")
                    continue

                if actual_time_col != 'future_timestamp_utc':
                    df.rename(columns={actual_time_col: 'future_timestamp_utc'}, inplace=True)
                    print(f"      Колонка '{actual_time_col}' переименована в 'future_timestamp_utc'.")

                df['future_timestamp_utc'] = pd.to_datetime(df['future_timestamp_utc']).dt.tz_localize('UTC')

                predicted_col_original_name = f'predicted_{target_column_name}'
                price_col_found = False
                actual_price_col = None

                if predicted_col_original_name in df.columns:
                    actual_price_col = predicted_col_original_name
                    price_col_found = True
                elif 'predicted_price' in df.columns:
                    actual_price_col = 'predicted_price'
                    price_col_found = True

                if not price_col_found:
                    print(
                        f"    Предупреждение: Колонка цены ('{predicted_col_original_name}' или 'predicted_price') не найдена в {file_path}. Пропуск файла.")
                    continue

                if actual_price_col != 'predicted_price':
                    df.rename(columns={actual_price_col: 'predicted_price'}, inplace=True)
                    print(f"      Колонка цены '{actual_price_col}' переименована в 'predicted_price'.")

                df['predicted_price'] = df['predicted_price'].apply(Decimal)

                # <ВАЛИДАЦИЯ 1.2: После переименования и преобразования типов>
                print(f"    Данные из файла {os.path.basename(file_path)} после обработки:")
                if JUPYTER_DISPLAY_AVAILABLE:
                    display(df[['future_timestamp_utc', 'predicted_price']].head(3))
                else:
                    print(df[['future_timestamp_utc', 'predicted_price']].head(3))
                print("    Типы данных после обработки:")
                print(df[['future_timestamp_utc', 'predicted_price']].dtypes)
                # </ВАЛИДАЦИЯ 1.2>

                model_type_dfs_to_concat.append(df[['future_timestamp_utc', 'predicted_price']])
                loaded_filenames_for_type.append(os.path.basename(file_path))
            except KeyError as e:
                print(
                    f"    Ошибка KeyError при обработке {file_path} для типа '{key_part}': {e}. Проверьте имена колонок.")
            except Exception as e:
                print(f"    Общая ошибка загрузки {file_path} для типа '{key_part}': {type(e).__name__} - {e}")

        if model_type_dfs_to_concat:
            combined_df_for_type = pd.concat(model_type_dfs_to_concat)
            combined_df_for_type.sort_values('future_timestamp_utc', inplace=True)
            combined_df_for_type.drop_duplicates(subset=['future_timestamp_utc'], keep='last', inplace=True)

            # <ВАЛИДАЦИЯ 1.3: После объединения файлов для типа модели>
            print(f"    --- Объединенный DataFrame для типа '{key_part}' ---")
            if JUPYTER_DISPLAY_AVAILABLE:
                display(combined_df_for_type.head())
            else:
                print(combined_df_for_type.head())
            print(f"    Размер: {combined_df_for_type.shape}")
            # print("    Информация о combined_df_for_type:")
            # combined_df_for_type.info(buf=sys.stdout)
            # </ВАЛИДАЦИЯ 1.3>

            all_predictions_data[key_part] = combined_df_for_type
            print(
                f"    Загружено и объединено {len(loaded_filenames_for_type)} файла(ов): {', '.join(loaded_filenames_for_type)}")
            print(f"    Итого {len(combined_df_for_type)} уникальных строк прогнозов для типа '{key_part}'.")
        else:
            print(f"    Не удалось загрузить ни одного валидного файла прогнозов для типа '{key_part}'.")
    return all_predictions_data


def get_sandbox_account_id(client: Client, deposit_amount: Decimal):
    try:
        accounts = client.sandbox.get_sandbox_accounts().accounts
        if accounts:
            account_id = accounts[0].id;
            print(f"Используется существующий счет в песочнице: {account_id}")
        else:
            account_id = client.sandbox.open_sandbox_account().account_id;
            print(
                f"Создан новый счет в песочнице: {account_id}")
        payin_amount_mv = create_money_value(deposit_amount, 'rub')
        client.sandbox.sandbox_pay_in(account_id=account_id, amount=payin_amount_mv)
        print(f"Счет {account_id} пополнен на {deposit_amount} RUB");
        return account_id
    except RequestError as e:
        print(f"Ошибка при работе со счетом в песочнице: {e}");
        return None


def get_current_price(client: Client, figi: str) -> Decimal | None:
    try:
        last_prices = client.market_data.get_last_prices(figi=[figi])
        if last_prices.last_prices: return quotation_to_decimal(last_prices.last_prices[0].price)
        print(f"Не удалось получить текущую цену для FIGI: {figi}");
        return None
    except RequestError as e:
        print(f"Ошибка API Tinkoff (цена): {e}");
        return None


def get_instrument_details(client: Client, figi: str):
    try:
        instrument_info = client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi)
        if instrument_info and instrument_info.instrument:
            return {"lot": instrument_info.instrument.lot,
                    "currency": instrument_info.instrument.currency,
                    "min_price_increment": quotation_to_decimal(instrument_info.instrument.min_price_increment),
                    "short_enabled_flag": instrument_info.instrument.short_enabled_flag}
        print(f"Не удалось получить инфо об инструменте FIGI: {figi}");
        return None
    except RequestError as e:
        print(f"Ошибка API Tinkoff (инструмент): {e}");
        return None


def place_sandbox_order(client: Client, figi: str, quantity_lots: int, direction: OrderDirection, account_id: str,
                        lot_size_param: int):
    if quantity_lots <= 0:
        print(f"  Попытка ордера отменена: некорректное количество лотов ({quantity_lots}).")
        return {"success": False, "message": "Incorrect lot quantity", "lots_executed": 0}

    print(f"  Попытка ордера: FIGI={figi}, Лоты={quantity_lots}, Направление={direction.name}")
    trade_result = {"success": False, "timestamp": now(), "order_id": None, "direction": direction,
                    "lots_requested": quantity_lots, "lots_executed": 0,
                    "executed_price_per_share": None, "total_order_amount_rub": None, "commission_rub": None,
                    "currency": "rub", "message": "",
                    "initial_cost_or_proceeds_rub": None}
    try:
        posted_order = client.sandbox.post_sandbox_order(figi=figi, quantity=quantity_lots, direction=direction,
                                                         account_id=account_id, order_type=OrderType.ORDER_TYPE_MARKET,
                                                         order_id="")

        trade_result["order_id"] = posted_order.order_id;
        trade_result["message"] = posted_order.message
        status_name = posted_order.execution_report_status.name
        print(
            f"  Ответ API (ID: {posted_order.order_id}): Статус={status_name}, Запрошено={posted_order.lots_requested}, Исполнено={posted_order.lots_executed}")
        trade_result["lots_executed"] = posted_order.lots_executed
        if posted_order.lots_executed > 0 and posted_order.execution_report_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
            trade_result["success"] = True
            if posted_order.executed_order_price:
                trade_result["executed_price_per_share"] = money_to_decimal(posted_order.executed_order_price)
                trade_result["currency"] = posted_order.executed_order_price.currency
                print(
                    f"    Цена исполнения (за 1 шт.): {trade_result['executed_price_per_share']} {trade_result['currency']}")
            else:
                print("    ВНИМАНИЕ: Ордер исполнен, но цена исполнения не указана.");
                trade_result["success"] = False
            if posted_order.total_order_amount: trade_result["total_order_amount_rub"] = money_to_decimal(
                posted_order.total_order_amount); print(
                f"    Общая сумма сделки (с комиссией): {trade_result['total_order_amount_rub']}")
            if posted_order.executed_commission: trade_result["commission_rub"] = money_to_decimal(
                posted_order.executed_commission); print(f"    Комиссия: {trade_result['commission_rub']}")

            if trade_result["executed_price_per_share"] and lot_size_param > 0:
                clean_value_of_trade = trade_result["executed_price_per_share"] * Decimal(
                    str(posted_order.lots_executed * lot_size_param))
                trade_result["initial_cost_or_proceeds_rub"] = clean_value_of_trade
            elif trade_result["success"]:
                print("    ВНИМАНИЕ: Не удалось рассчитать чистую стоимость сделки (initial_cost_or_proceeds_rub).")


        elif posted_order.execution_report_status != OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
            trade_result["success"] = False
            trade_result["message"] = f"Ордер не исполнен полностью. Статус: {status_name}. {posted_order.message}"
            print(f"    {trade_result['message']}")

        return trade_result
    except RequestError as e:
        print(f"Ошибка API Tinkoff при размещении ордера: {e}");
        trade_result["message"] = str(e);
        return trade_result


def plot_prices_and_trades(market_price_history, trade_log, predicted_price_history, figi_main, output_dir="."):
    if not market_price_history and not predicted_price_history and not trade_log:
        print("Нет данных для построения графика цен, прогнозов и сделок.")
        return

    plt.figure(figsize=(15, 7))

    if market_price_history:
        timestamps_market_utc, prices_market = zip(*market_price_history)
        prices_market_float = [float(p) for p in prices_market]
        timestamps_market_msk = [ts.astimezone(MOSCOW_TZ) for ts in timestamps_market_utc]
        plt.plot(timestamps_market_msk, prices_market_float, label="Рыночная цена", color="blue", alpha=0.7, zorder=1)

    if predicted_price_history:
        timestamps_pred_utc, prices_pred_decimal = zip(*predicted_price_history)
        prices_pred_float = [float(p) for p in prices_pred_decimal]
        timestamps_pred_msk = [ts.astimezone(MOSCOW_TZ) for ts in timestamps_pred_utc]
        plt.plot(timestamps_pred_msk, prices_pred_float, label="Спрогнозированная цена (исп. для решения)",
                 color="orange", linestyle='--', marker='.', markersize=5, alpha=0.8, zorder=1)

    buy_times_msk = [t["timestamp"].astimezone(MOSCOW_TZ) for t in trade_log if
                     t["direction_str"].startswith("Покупка") and t["success"] and t[
                         "executed_price_per_share"] is not None]
    buy_prices_float = [float(t["executed_price_per_share"]) for t in trade_log if
                        t["direction_str"].startswith("Покупка") and t["success"] and t[
                            "executed_price_per_share"] is not None]

    sell_times_msk = [t["timestamp"].astimezone(MOSCOW_TZ) for t in trade_log if
                      t["direction_str"].startswith("Продажа") and t["success"] and t[
                          "executed_price_per_share"] is not None]
    sell_prices_float = [float(t["executed_price_per_share"]) for t in trade_log if
                         t["direction_str"].startswith("Продажа") and t["success"] and t[
                             "executed_price_per_share"] is not None]

    if buy_times_msk and buy_prices_float:
        plt.scatter(buy_times_msk, buy_prices_float, label="Покупки/Закр. шорта", color="green",
                    marker="^", s=120, edgecolors='black', zorder=2)
    if sell_times_msk and sell_prices_float:
        plt.scatter(sell_times_msk, sell_prices_float, label="Продажи/Откр. шорта",
                    color="red", marker="v", s=120, edgecolors='black', zorder=2)

    plt.title(f"График цен для {figi_main}: Рыночные цены, прогнозы и сделки бота (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Цена", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=MOSCOW_TZ))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"price_trades_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График цен сохранен: {filename}")

    try:
        if JUPYTER_DISPLAY_AVAILABLE:  # В Jupyter можно показать график прямо в выводе
            plt.show()
        else:  # Для обычного скрипта
            plt.show(block=False)
            plt.pause(1)
        plt.close()
    except Exception as e:
        print(f"Ошибка при отображении/закрытии графика цен: {e}")


def plot_profitability(portfolio_value_history, initial_balance_rub, figi_main, output_dir="."):
    if not portfolio_value_history or len(portfolio_value_history) < 2:
        print("Недостаточно данных о доходности для графика.")
        return

    timestamps_utc, values_decimal = zip(*portfolio_value_history)
    values_float = [float(v) for v in values_decimal]
    initial_balance_rub_float = float(initial_balance_rub)

    timestamps_msk = [ts.astimezone(MOSCOW_TZ) for ts in timestamps_utc]
    if initial_balance_rub_float == 0:
        print("Начальный баланс равен нулю, невозможно рассчитать доходность.")
        return

    profitability_percent = [((v / initial_balance_rub_float) - 1) * 100 for v in values_float]

    plt.figure(figsize=(15, 7))
    plt.plot(timestamps_msk, profitability_percent, label="Доходность портфеля", color="purple", marker='.')
    plt.axhline(0, color='grey', linestyle='--', lw=1)
    plt.title(f"График доходности портфеля для {figi_main} (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Доходность, %", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=MOSCOW_TZ))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"profitability_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График доходности сохранен: {filename}")

    try:
        if JUPYTER_DISPLAY_AVAILABLE:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
        plt.close()
    except Exception as e:
        print(f"Ошибка при отображении/закрытии графика доходности: {e}")


# --- НОВАЯ ФУНКЦИЯ ДЛЯ ГРАФИКА СООТНОШЕНИЯ ПРОГНОЗ/РЕАЛЬНАЯ ЦЕНА ---
def plot_forecast_real_ratio(timestamps_utc, actual_prices: list[Decimal], predicted_prices: list[Decimal], figi_main,
                             output_dir="."):
    if not timestamps_utc or len(timestamps_utc) < MIN_SAMPLES_FOR_METRICS:
        print("Недостаточно данных для построения графика соотношения Прогноз/Реальная цена.")
        return

    ratios = []
    valid_timestamps_utc = []
    for i in range(len(timestamps_utc)):
        actual = actual_prices[i]
        predicted = predicted_prices[i]
        if actual is not None and predicted is not None and actual != Decimal(0):
            try:
                ratio = predicted / actual
                ratios.append(float(ratio))  # Сохраняем как float для графика
                valid_timestamps_utc.append(timestamps_utc[i])
            except (DivisionByZero, TypeError):
                pass  # Пропускаем, если деление на ноль или тип несовместим
        elif actual == Decimal(0):
            print(
                f"Предупреждение: Реальная цена была 0 в {format_datetime_msk(timestamps_utc[i])}, точка для графика соотношения пропущена.")

    if not ratios:
        print("Нет валидных данных для построения графика соотношения.")
        return

    timestamps_msk = [ts.astimezone(MOSCOW_TZ) for ts in valid_timestamps_utc]

    plt.figure(figsize=(15, 7))
    plt.plot(timestamps_msk, ratios, label="Соотношение Прогноз/Реальная цена", color="teal", marker='.')
    plt.axhline(1, color='red', linestyle='--', lw=1, label="Идеальное совпадение (Прогноз/Реал = 1)")

    plt.title(f"График соотношения Прогноз/Реальная цена для {figi_main} (Время MSK)", fontsize=16)
    plt.xlabel("Время (MSK)", fontsize=12)
    plt.ylabel("Соотношение (Прогноз / Реальная цена)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=MOSCOW_TZ))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"forecast_real_ratio_chart_{figi_main}.png")
    plt.savefig(filename)
    print(f"График соотношения Прогноз/Реальная цена сохранен: {filename}")

    try:
        if JUPYTER_DISPLAY_AVAILABLE:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(1)
        plt.close()
    except Exception as e:
        print(f"Ошибка при отображении/закрытии графика соотношения: {e}")


def generate_pnl_table_df(trade_log_enriched, figi):
    if not trade_log_enriched: print("Нет данных для таблицы P&L."); return pd.DataFrame()
    df_pnl = pd.DataFrame(trade_log_enriched).copy()
    df_pnl["Время (MSK)"] = df_pnl["timestamp"].apply(format_datetime_msk)
    df_pnl = df_pnl.rename(
        columns={"trade_number": "№", "figi": "FIGI", "direction_str": "Операция", "lots_executed": "Лоты",
                 "shares_executed": "Акции",
                 "executed_price_per_share_str": "Цена за акцию",
                 "initial_cost_or_proceeds_rub_str": "Стоимость/Выручка (без ком.)",
                 "commission_rub_str": "Комиссия", "realized_pnl_this_trade_str": "P&L сделки",
                 "cash_after_trade_str": "Кэш после сделки",
                 "portfolio_value_after_trade_str": "Портфель после сделки"})
    columns_to_show = ["№", "Время (MSK)", "FIGI", "Операция", "Лоты", "Акции", "Цена за акцию",
                       "Стоимость/Выручка (без ком.)",
                       "Комиссия", "P&L сделки", "Кэш после сделки", "Портфель после сделки"]
    for col in columns_to_show:
        if col not in df_pnl.columns: df_pnl[col] = '-'
    return df_pnl[columns_to_show]


def check_if_future_predictions_exist(loaded_predictions_by_type, current_time_utc):
    if not loaded_predictions_by_type: return False
    for key_part, preds_df in loaded_predictions_by_type.items():
        if not preds_df[preds_df['future_timestamp_utc'] > current_time_utc].empty:
            return True
    return False


def trading_bot_main():
    print("--- Запуск торгового робота в песочнице Tinkoff ---")
    trade_log_enriched = []
    market_price_history = []
    portfolio_value_history = []
    predicted_price_history = []

    actual_prices_for_metrics_float = []  # Храним как float для sklearn
    predictions_for_metrics_float = []  # Храним как float для sklearn
    timestamps_for_metrics_utc = []

    current_positions = {
        INSTRUMENT_FIGI: {
            'quantity_shares': Decimal(0),
            'average_entry_price_per_share': Decimal(0),
        }
    }
    current_market_price: Decimal | None = None
    loaded_predictions_by_type = load_predictions(MODEL_NAME_KEY_PARTS, OUTPUT_CSV_DIRECTORY, INSTRUMENT_FIGI,
                                                  FEATURE_COLUMN, NUM_FILES_TO_LOAD_PER_MODEL_TYPE)

    # <ВАЛИДАЦИЯ 1.4: Итоговый словарь loaded_predictions_by_type>
    print("\n--- Итоговые загруженные прогнозы по типам моделей (валидация в trading_bot_main) ---")
    if loaded_predictions_by_type:
        for model_key, df_preds in loaded_predictions_by_type.items():
            print(f"  Модель: {model_key}")
            print(f"    Форма: {df_preds.shape}")
            if JUPYTER_DISPLAY_AVAILABLE:
                display(df_preds.head())
            else:
                print(df_preds.head())
    else:
        print("  Прогнозы не загружены.")
    # </ВАЛИДАЦИЯ 1.4>

    if not loaded_predictions_by_type:
        print("Нет загруженных прогнозов ни для одного типа моделей. Робот не может продолжить работу.")
        return

    user_stop_requested = False
    stop_loss_triggered_reason = None  # Причина остановки по стоп-лоссу или другим условиям

    with Client(TINKOFF_SANDBOX_TOKEN) as client:
        sandbox_account_id = get_sandbox_account_id(client, INITIAL_DEPOSIT_RUB)
        if not sandbox_account_id: return
        current_cash_rub = INITIAL_DEPOSIT_RUB
        portfolio_value_history.append((datetime.now(timezone.utc) - timedelta(seconds=10), INITIAL_DEPOSIT_RUB))

        instrument_details = get_instrument_details(client, INSTRUMENT_FIGI)
        if not instrument_details: return
        lot_size = instrument_details['lot']
        instrument_currency = instrument_details['currency']
        short_enabled = instrument_details['short_enabled_flag']
        print(
            f"Детали {INSTRUMENT_FIGI}: Лот={lot_size}, Валюта={instrument_currency}, Шаг цены={instrument_details['min_price_increment']}, Шорт разрешен: {short_enabled}")

        operations_count = 0
        trade_number_counter = 0

        last_seen_market_price: Decimal | None = None
        last_price_change_time_utc: datetime | None = None
        deviation_start_time_utc = None

        while not user_stop_requested:
            operations_count += 1
            current_time_in_moscow = datetime.now(MOSCOW_TZ)
            current_cycle_time_utc = current_time_in_moscow.astimezone(timezone.utc)

            print(f"\n--- Цикл {operations_count} ({current_time_in_moscow.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

            if operations_count == 100:  # Используем operations_count # ИЗМЕНЕНО ДЛЯ ОТЛАДКИ
                print("!!! ОТЛАДКА: Достигнут N-й цикл. Прерывание работы программы. !!!")
                user_stop_requested = True
                stop_loss_triggered_reason = f"Отладочное прерывание на {operations_count}-м цикле."

            if user_stop_requested:  # Проверка перед основной логикой цикла
                break

            if not check_if_future_predictions_exist(loaded_predictions_by_type, current_cycle_time_utc):
                print("Больше нет будущих прогнозов ни в одном из типов файлов. Завершение работы.")
                stop_loss_triggered_reason = "Закончились будущие прогнозы."
                user_stop_requested = True

            if user_stop_requested:
                break

            current_market_price = get_current_price(client, INSTRUMENT_FIGI)
            if current_market_price is None:
                print("  Не удалось получить текущую цену. Пропуск этой итерации.")
                print(f"  Ожидание {TRADE_INTERVAL_SECONDS // 60} мин перед следующей попыткой...")
                time.sleep(TRADE_INTERVAL_SECONDS)
                continue

            print(f"  Текущая рыночная цена: {current_market_price:.4f}")
            market_price_history.append((current_cycle_time_utc, current_market_price))
            # <ВАЛИДАЦИЯ 2.1: Данные о текущей рыночной цене и истории цен>
            if market_price_history:
                print("  Последние рыночные цены (хвост):")
                temp_market_df = pd.DataFrame(market_price_history, columns=['timestamp_utc', 'market_price'])
                temp_market_df['timestamp_msk'] = temp_market_df['timestamp_utc'].apply(format_datetime_msk)
                if JUPYTER_DISPLAY_AVAILABLE:
                    display(temp_market_df[['timestamp_msk', 'market_price']].tail(3))
                else:
                    print(temp_market_df[['timestamp_msk', 'market_price']].tail(3))
            # </ВАЛИДАЦИЯ 2.1>

            if last_seen_market_price is None:
                last_seen_market_price = current_market_price
                last_price_change_time_utc = current_cycle_time_utc
            elif current_market_price != last_seen_market_price:
                last_seen_market_price = current_market_price
                last_price_change_time_utc = current_cycle_time_utc
                print(
                    f"  Цена изменилась. Новая цена: {current_market_price:.4f}. Время последнего изменения: {format_datetime_msk(last_price_change_time_utc)}")
            else:
                if last_price_change_time_utc:
                    time_since_last_change = current_cycle_time_utc - last_price_change_time_utc
                    print(
                        f"  Цена не изменилась ({current_market_price:.4f}). Время без изменений: {time_since_last_change.total_seconds() / 60:.2f} мин.")
                    if time_since_last_change >= timedelta(minutes=PRICE_STAGNATION_TIMEOUT_MINUTES):
                        print(
                            f"!!! ЦЕНА НЕ МЕНЯЛАСЬ БОЛЕЕ {PRICE_STAGNATION_TIMEOUT_MINUTES} МИНУТ. ЗАВЕРШЕНИЕ ТОРГОВЛИ. !!!")
                        stop_loss_triggered_reason = f"Цена не менялась более {PRICE_STAGNATION_TIMEOUT_MINUTES} минут."
                        user_stop_requested = True

            if user_stop_requested:
                break

            prediction_search_start_utc = current_cycle_time_utc
            prediction_search_end_utc = current_cycle_time_utc + timedelta(minutes=PREDICTION_LOOKAHEAD_MINUTES)
            target_prediction_time_for_selection_utc = prediction_search_end_utc

            model_type_target_prices = []
            print(f"  Анализ прогнозов по типам моделей на следующие {PREDICTION_LOOKAHEAD_MINUTES} минут.")
            print(
                f"  Окно поиска UTC: ({prediction_search_start_utc.strftime('%Y-%m-%d %H:%M:%S')}, {prediction_search_end_utc.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(
                f"  Окно поиска MSK: ({format_datetime_msk(prediction_search_start_utc)}, {format_datetime_msk(prediction_search_end_utc)}]")

            for key_part, preds_df_original in loaded_predictions_by_type.items():
                relevant_preds_df = preds_df_original[
                    (preds_df_original['future_timestamp_utc'] > prediction_search_start_utc) &
                    (preds_df_original['future_timestamp_utc'] <= prediction_search_end_utc)
                    ]
                # <ВАЛИДАЦИЯ 2.2: Отфильтрованные прогнозы (relevant_preds_df)>
                print(
                    f"    Для типа '{key_part}', релевантные прогнозы в окне ({format_datetime_msk(prediction_search_start_utc)} - {format_datetime_msk(prediction_search_end_utc)}):")
                if not relevant_preds_df.empty:
                    if JUPYTER_DISPLAY_AVAILABLE:
                        display(relevant_preds_df.head(3))
                    else:
                        print(relevant_preds_df.head(3))
                    print(f"    Найдено релевантных: {relevant_preds_df.shape[0]}")
                else:
                    print("     Нет релевантных прогнозов.")
                # </ВАЛИДАЦИЯ 2.2>

                if not relevant_preds_df.empty:
                    relevant_preds_df = relevant_preds_df.copy()
                    relevant_preds_df.loc[:, 'time_diff'] = (relevant_preds_df[
                                                                 'future_timestamp_utc'] - target_prediction_time_for_selection_utc).abs()
                    closest_pred = relevant_preds_df.sort_values(by='time_diff').iloc[0]
                    # <ВАЛИДАЦИЯ 2.3: Выбранный ближайший прогноз (closest_pred)>
                    print(f"    Ближайший прогноз для '{key_part}':")
                    # .T транспонирует Series в строку DataFrame для красивого вывода
                    if JUPYTER_DISPLAY_AVAILABLE:
                        display(pd.DataFrame(closest_pred).T)
                    else:
                        print(pd.DataFrame(closest_pred).T)
                    # </ВАЛИДАЦИЯ 2.3>
                    target_price = closest_pred['predicted_price']
                    pred_time_log_utc = closest_pred['future_timestamp_utc']
                    model_type_target_prices.append(target_price)
                    print(
                        f"    Тип '{key_part}': найден прогноз {target_price:.4f} на {format_datetime_msk(pred_time_log_utc)}")
                else:
                    print(
                        f"    Тип '{key_part}': нет прогнозов в окне на следующие {PREDICTION_LOOKAHEAD_MINUTES} минут.")

            overall_target_predicted_price: Decimal | None = None
            if not model_type_target_prices:
                print(
                    f"  Нет прогнозов на ~{PREDICTION_LOOKAHEAD_MINUTES} мин. вперед ни от одного типа моделей. Пропуск торговой операции.")
            else:
                sum_of_predictions = sum(model_type_target_prices)
                count_of_predictions = Decimal(len(model_type_target_prices))

                if count_of_predictions > Decimal(0):
                    overall_target_predicted_price = sum_of_predictions / count_of_predictions
                else:
                    overall_target_predicted_price = None

                if overall_target_predicted_price is not None:
                    print(
                        f"  Итоговый средний прогноз на ~{PREDICTION_LOOKAHEAD_MINUTES} мин: {overall_target_predicted_price:.4f}")
                    predicted_price_history.append((current_cycle_time_utc, overall_target_predicted_price))
                    # <ВАЛИДАЦИЯ 2.4: Итоговый усредненный прогноз и история прогнозов>
                    if predicted_price_history:
                        print("  Последние использованные прогнозы (хвост):")
                        temp_pred_df = pd.DataFrame(predicted_price_history,
                                                    columns=['timestamp_utc', 'predicted_price'])
                        temp_pred_df['timestamp_msk'] = temp_pred_df['timestamp_utc'].apply(format_datetime_msk)
                        if JUPYTER_DISPLAY_AVAILABLE:
                            display(temp_pred_df[['timestamp_msk', 'predicted_price']].tail(3))
                        else:
                            print(temp_pred_df[['timestamp_msk', 'predicted_price']].tail(3))
                    # </ВАЛИДАЦИЯ 2.4>

                    if current_market_price is not None and overall_target_predicted_price is not None:
                        actual_prices_for_metrics_float.append(float(current_market_price))
                        predictions_for_metrics_float.append(float(overall_target_predicted_price))
                        timestamps_for_metrics_utc.append(current_cycle_time_utc)

                        if SKLEARN_AVAILABLE and len(actual_prices_for_metrics_float) >= MIN_SAMPLES_FOR_METRICS:
                            # <ВАЛИДАЦИЯ 2.5: Данные для расчета метрик>
                            print("  --- Данные для расчета метрик (хвост) ---")
                            metrics_df = pd.DataFrame({
                                'timestamp_utc': timestamps_for_metrics_utc,
                                'actual_price': actual_prices_for_metrics_float,
                                'predicted_price': predictions_for_metrics_float
                            })
                            metrics_df['timestamp_msk'] = metrics_df['timestamp_utc'].apply(format_datetime_msk)
                            if JUPYTER_DISPLAY_AVAILABLE:
                                display(metrics_df[['timestamp_msk', 'actual_price', 'predicted_price']].tail())
                            else:
                                print(metrics_df[['timestamp_msk', 'actual_price', 'predicted_price']].tail())
                            # </ВАЛИДАЦИЯ 2.5>
                            try:
                                mse = mean_squared_error(actual_prices_for_metrics_float, predictions_for_metrics_float)
                                mae = mean_absolute_error(actual_prices_for_metrics_float,
                                                          predictions_for_metrics_float)
                                r2 = r2_score(actual_prices_for_metrics_float, predictions_for_metrics_float)
                                print(
                                    f"  Метрики точности прогноза (на {len(actual_prices_for_metrics_float)} точках):")
                                print(f"    MSE: {mse:.4f}")
                                print(f"    MAE: {mae:.4f}")
                                print(f"    R^2: {r2:.4f}")

                                if Decimal(str(r2)) < R2_ACCURACY_THRESHOLD_STOP:
                                    print(
                                        f"  ПРЕДУПРЕЖДЕНИЕ: Точность R2 ({r2:.4f}) ниже порога ({R2_ACCURACY_THRESHOLD_STOP}). "
                                        f"Робот НЕ будет остановлен по этой причине.")
                            except Exception as e_metrics:
                                print(f"  Ошибка при расчете метрик: {e_metrics}")
                        elif not SKLEARN_AVAILABLE:
                            print("  Расчет метрик пропущен: scikit-learn недоступна.")
                        elif len(actual_prices_for_metrics_float) < MIN_SAMPLES_FOR_METRICS:
                            print(
                                f"  Недостаточно данных для расчета метрик ({len(actual_prices_for_metrics_float)}/{MIN_SAMPLES_FOR_METRICS}).")

                    if not user_stop_requested:
                        if overall_target_predicted_price == Decimal(0):
                            print(
                                "  ПРЕДУПРЕЖДЕНИЕ: Спрогнозированная цена равна 0. Проверка стоп-лоссов по прогнозу пропускается в этом цикле.")
                        else:
                            try:
                                price_deviation_percent = abs(
                                    current_market_price - overall_target_predicted_price) / overall_target_predicted_price * Decimal(
                                    100)
                                print(
                                    f"  Отклонение реальной цены ({current_market_price:.4f}) от прогноза ({overall_target_predicted_price:.4f}): {price_deviation_percent:.2f}%")

                                if price_deviation_percent > INSTANT_DEVIATION_THRESHOLD_PERCENT:
                                    stop_loss_triggered_reason = (
                                        f"ЖЕСТКИЙ СТОП-ЛОСС: Цена отклонилась на {price_deviation_percent:.2f}% "
                                        f"(> {INSTANT_DEVIATION_THRESHOLD_PERCENT}%) от прогноза. "
                                        f"Реальная: {current_market_price:.4f}, Прогноз: {overall_target_predicted_price:.4f}."
                                    )
                                    print(f"!!! {stop_loss_triggered_reason} ОСТАНОВКА РОБОТА. !!!")
                                    user_stop_requested = True

                                if not user_stop_requested:
                                    if price_deviation_percent > CONTINUOUS_DEVIATION_THRESHOLD_PERCENT:
                                        if deviation_start_time_utc is None:
                                            deviation_start_time_utc = current_cycle_time_utc
                                            print(
                                                f"  ВНИМАНИЕ: Начало периода отклонения цены от прогноза > {CONTINUOUS_DEVIATION_THRESHOLD_PERCENT}%. "
                                                f"Таймер мягкого стоп-лосса ({CONTINUOUS_DEVIATION_DURATION_MINUTES} мин) запущен.")

                                        time_since_deviation_start = current_cycle_time_utc - deviation_start_time_utc
                                        print(
                                            f"    Время непрерывного отклонения: {time_since_deviation_start.total_seconds() / 60:.1f} мин.")
                                        if time_since_deviation_start >= timedelta(
                                                minutes=CONTINUOUS_DEVIATION_DURATION_MINUTES):
                                            stop_loss_triggered_reason = (
                                                f"МЯГКИЙ СТОП-ЛОСС: Цена непрерывно отклонялась более чем на {CONTINUOUS_DEVIATION_THRESHOLD_PERCENT}% "
                                                f"от прогноза на протяжении {time_since_deviation_start.total_seconds() / 60:.1f} минут "
                                                f"(лимит {CONTINUOUS_DEVIATION_DURATION_MINUTES} мин). "
                                                f"Текущее отклонение: {price_deviation_percent:.2f}%. "
                                                f"Реальная: {current_market_price:.4f}, Прогноз: {overall_target_predicted_price:.4f}."
                                            )
                                            print(f"!!! {stop_loss_triggered_reason} ОСТАНОВКА РОБОТА. !!!")
                                            user_stop_requested = True
                                    else:
                                        if deviation_start_time_utc is not None:
                                            print(
                                                f"  ИНФО: Отклонение цены от прогноза вернулось в норму (<={CONTINUOUS_DEVIATION_THRESHOLD_PERCENT}%). "
                                                "Сброс таймера мягкого стоп-лосса.")
                                        deviation_start_time_utc = None
                            except DivisionByZero:
                                print(
                                    "  ПРЕДУПРЕЖДЕНИЕ: Деление на ноль при расчете отклонения цены (прогноз равен 0). Пропуск проверки стоп-лосса.")
                            except Exception as e_stoploss:
                                print(f"  ОШИБКА в блоке стоп-лоссов: {type(e_stoploss).__name__} - {e_stoploss}")

            if user_stop_requested:
                break

            if overall_target_predicted_price is not None and current_market_price is not None:
                pos_data = current_positions[INSTRUMENT_FIGI]
                current_shares_qty = pos_data['quantity_shares']

                target_shares_qty = TARGET_NO_POSITION_SHARES
                action_reason = ""
                order_direction = None

                if overall_target_predicted_price > current_market_price:
                    if current_market_price > Decimal(0) and lot_size > 0:
                        cost_per_lot = current_market_price * Decimal(lot_size)
                        if cost_per_lot > Decimal(0) and current_cash_rub >= cost_per_lot:
                            max_lots_can_buy = int(current_cash_rub // cost_per_lot)
                            target_shares_qty = Decimal(max_lots_can_buy * lot_size)
                            action_reason = f"Ожидается рост цены. Покупка на доступные средства ({max_lots_can_buy} лот(ов))."
                        else:
                            target_shares_qty = TARGET_NO_POSITION_SHARES
                            action_reason = "Ожидается рост цены, но недостаточно средств для покупки 1 лота или цена/лот некорректны. Цель - нет позиции."
                    else:
                        target_shares_qty = TARGET_NO_POSITION_SHARES
                        action_reason = "Ожидается рост цены, но цена/лот некорректны. Цель - нет позиции."

                elif overall_target_predicted_price < current_market_price:
                    if short_enabled:
                        if current_market_price > Decimal(0) and lot_size > 0:
                            cost_per_lot = current_market_price * Decimal(lot_size)
                            if cost_per_lot > Decimal(0):
                                current_equity = current_cash_rub
                                if current_shares_qty < 0:  # Если уже в шорте
                                    # Примерный расчет эквити. Для точного нужен учет маржи.
                                    # Стоимость откупа текущего шорта
                                    cost_to_cover = abs(current_shares_qty) * current_market_price
                                    # Прибыль/убыток от шорта
                                    pnl_from_short = abs(current_shares_qty) * pos_data[
                                        'average_entry_price_per_share'] - cost_to_cover
                                    # Эквити с учетом открытого шорта
                                    current_equity = current_cash_rub + pnl_from_short

                                max_short_value = min(INITIAL_DEPOSIT_RUB, max(Decimal(0), current_equity))
                                max_lots_for_short_limit = int(max_short_value // cost_per_lot)

                                if max_lots_for_short_limit > 0:
                                    target_short_shares = Decimal(-max_lots_for_short_limit * lot_size)
                                    if current_shares_qty > 0:  # Если был лонг
                                        target_shares_qty = target_short_shares
                                        action_reason = f"Ожидается падение. Закрытие лонга ({current_shares_qty} акций) и открытие шорта до {target_short_shares} акций ({max_lots_for_short_limit} лот(ов))."
                                    else:  # Если был 0 или уже шорт
                                        target_shares_qty = target_short_shares
                                        action_reason = f"Ожидается падение цены. Шорт/увеличение шорта до {target_short_shares} акций ({max_lots_for_short_limit} лот(ов))."
                                else:  # Не хватает лимита на открытие/увеличение шорта
                                    if current_shares_qty > 0:  # Если был лонг
                                        target_shares_qty = TARGET_NO_POSITION_SHARES
                                        action_reason = f"Ожидается падение цены. Шорт-лимит ({max_short_value:.2f}) не позволяет открыть новую позицию. Закрытие лонга ({current_shares_qty} акций)."
                                    else:  # Если был 0 или уже шорт, но лимит не позволяет увеличить
                                        target_shares_qty = current_shares_qty
                                        action_reason = f"Ожидается падение цены, но шорт-лимит ({max_short_value:.2f}) не позволяет открыть/увеличить позицию. Цель - текущая позиция ({current_shares_qty} акций)."
                            else:  # cost_per_lot <= 0
                                target_shares_qty = TARGET_NO_POSITION_SHARES if current_shares_qty > 0 else current_shares_qty
                                action_reason = "Ожидается падение цены, но цена/лот некорректны. Если лонг - закрыть, иначе - текущая позиция."
                        else:  # current_market_price <= 0 or lot_size <= 0
                            target_shares_qty = TARGET_NO_POSITION_SHARES if current_shares_qty > 0 else current_shares_qty
                            action_reason = "Ожидается падение цены, но цена/лот некорректны. Если лонг - закрыть, иначе - текущая позиция."
                    else:  # short_enabled is False
                        target_shares_qty = TARGET_NO_POSITION_SHARES
                        action_reason = "Ожидается падение цены, но шорт запрещен. Цель - нет позиции (закрытие лонга если есть)."

                else:  # overall_target_predicted_price == current_market_price
                    action_reason = f"Прогноз ({overall_target_predicted_price:.4f}) ~ Цена ({current_market_price:.4f}). Цель - нет позиции."
                    target_shares_qty = TARGET_NO_POSITION_SHARES

                shares_to_trade = target_shares_qty - current_shares_qty
                trade_action_details = None

                if shares_to_trade == Decimal(0):
                    print(
                        f"  РЕШЕНИЕ: Текущая позиция ({current_shares_qty} акций) соответствует целевой. {action_reason} Пропуск.")
                else:
                    if shares_to_trade > Decimal(0):
                        order_direction = OrderDirection.ORDER_DIRECTION_BUY
                        print(
                            f"  РЕШЕНИЕ: Купить {shares_to_trade} акций. Цель: {target_shares_qty}, Сейчас: {current_shares_qty}. {action_reason}")
                    else:
                        order_direction = OrderDirection.ORDER_DIRECTION_SELL
                        print(
                            f"  РЕШЕНИЕ: Продать {abs(shares_to_trade)} акций. Цель: {target_shares_qty}, Сейчас: {current_shares_qty}. {action_reason}")

                    lots_to_trade_abs = int(abs(shares_to_trade) / Decimal(lot_size))

                    if Decimal(lots_to_trade_abs * lot_size) != abs(shares_to_trade):
                        if abs(shares_to_trade) % Decimal(lot_size) != Decimal(0):
                            print(
                                f" ПРЕДУПРЕЖДЕНИЕ: Количество акций для сделки ({abs(shares_to_trade)}) было не кратно лоту ({lot_size}). "
                                f"Будет предпринята попытка торговать {lots_to_trade_abs} лот(ов) ({lots_to_trade_abs * lot_size} акций)."
                            )
                        if lots_to_trade_abs == 0 and abs(shares_to_trade) > Decimal(0):
                            print(
                                f"   Сделка невозможна, так как требуемое изменение позиции ({abs(shares_to_trade)} акций) меньше размера лота ({lot_size}).")

                    if lots_to_trade_abs > 0:
                        trade_action_details = place_sandbox_order(client, INSTRUMENT_FIGI, lots_to_trade_abs,
                                                                   order_direction, sandbox_account_id, lot_size)
                        # <ВАЛИДАЦИЯ 2.6: Информация о сделке (trade_action_details)>
                        print("  --- Результат размещения ордера (trade_action_details) ---")
                        if trade_action_details:
                            if JUPYTER_DISPLAY_AVAILABLE:
                                display(pd.DataFrame([trade_action_details]))
                            else:
                                print(pd.DataFrame([trade_action_details]))
                        else:
                            print("  Нет деталей по сделке (trade_action_details is None)")
                        # </ВАЛИДАЦИЯ 2.6>
                    else:
                        if abs(shares_to_trade) > Decimal(0):
                            print(
                                f"  Сделка отменена: количество лотов для торговли равно 0 (требовалось {shares_to_trade} акций, что меньше 1 лота).")

                if trade_action_details and trade_action_details["success"] and trade_action_details[
                    "lots_executed"] > 0:
                    trade_number_counter += 1
                    enriched_trade_info = trade_action_details.copy()
                    enriched_trade_info["trade_number"] = trade_number_counter
                    enriched_trade_info["figi"] = INSTRUMENT_FIGI
                    enriched_trade_info["timestamp"] = trade_action_details["timestamp"]

                    executed_lots = trade_action_details["lots_executed"]
                    shares_actually_traded_abs = Decimal(executed_lots * lot_size)
                    price_per_share = trade_action_details["executed_price_per_share"]
                    commission_rub = trade_action_details.get("commission_rub", Decimal(0))

                    initial_value_of_traded_shares = trade_action_details.get("initial_cost_or_proceeds_rub",
                                                                              Decimal(0))
                    if initial_value_of_traded_shares == Decimal(0) and price_per_share:  # подстраховка
                        initial_value_of_traded_shares = price_per_share * shares_actually_traded_abs

                    realized_pnl_this_trade = Decimal(0)

                    prev_qty = pos_data['quantity_shares']
                    prev_avg_price = pos_data['average_entry_price_per_share']

                    if trade_action_details["direction"] == OrderDirection.ORDER_DIRECTION_BUY:
                        current_cash_rub -= (initial_value_of_traded_shares + commission_rub)
                        if prev_qty < Decimal(0):  # Покупка для закрытия/уменьшения шорта
                            enriched_trade_info["direction_str"] = "Покупка (закр/уменьш. шорт)"
                            shares_to_cover = min(shares_actually_traded_abs, abs(prev_qty))
                            realized_pnl_this_trade = (
                                                                  prev_avg_price - price_per_share) * shares_to_cover  # Прибыль если цена откупа < цены продажи шорта

                            pos_data['quantity_shares'] += shares_to_cover  # Уменьшаем абсолютное значение шорта
                            if pos_data['quantity_shares'] == Decimal(0):
                                pos_data['average_entry_price_per_share'] = Decimal(0)
                            # Если средняя цена шорта остается той же - это ОК, т.к. часть позиции закрыта

                            if shares_actually_traded_abs > shares_to_cover:  # Если купили больше, чем было в шорте -> открыли лонг
                                remaining_shares_bought = shares_actually_traded_abs - shares_to_cover
                                pos_data['quantity_shares'] += remaining_shares_bought  # Становится положительным
                                pos_data[
                                    'average_entry_price_per_share'] = price_per_share  # Новая цена входа для лонга
                                enriched_trade_info["direction_str"] = "Покупка (закр.шорт, откр.лонг)"
                            print(
                                f"    Закрыто/уменьшено в шорте: {shares_to_cover} акций. P&L (без ком.): {realized_pnl_this_trade:.2f}")

                        else:  # Покупка для открытия/увеличения лонга
                            enriched_trade_info["direction_str"] = "Покупка (лонг)"
                            new_total_cost = (prev_avg_price * prev_qty) + initial_value_of_traded_shares
                            pos_data['quantity_shares'] += shares_actually_traded_abs
                            pos_data['average_entry_price_per_share'] = new_total_cost / pos_data['quantity_shares'] if \
                                pos_data['quantity_shares'] > Decimal(0) else Decimal(0)
                            print(f"    Открыт/увеличен лонг: {shares_actually_traded_abs} акций.")
                        realized_pnl_this_trade -= commission_rub  # Учитываем комиссию в P&L

                    elif trade_action_details["direction"] == OrderDirection.ORDER_DIRECTION_SELL:
                        current_cash_rub += (initial_value_of_traded_shares - commission_rub)
                        if prev_qty > Decimal(0):  # Продажа для закрытия/уменьшения лонга
                            enriched_trade_info["direction_str"] = "Продажа (закр/уменьш. лонг)"
                            shares_to_sell_from_long = min(shares_actually_traded_abs, prev_qty)
                            realized_pnl_this_trade = (
                                                                  price_per_share - prev_avg_price) * shares_to_sell_from_long  # Прибыль если цена продажи > цены покупки лонга

                            pos_data['quantity_shares'] -= shares_to_sell_from_long
                            if pos_data['quantity_shares'] == Decimal(0):
                                pos_data['average_entry_price_per_share'] = Decimal(0)
                            # Если средняя цена лонга остается той же - это ОК

                            if shares_actually_traded_abs > shares_to_sell_from_long:  # Если продали больше, чем было в лонге -> открыли шорт
                                remaining_shares_sold = shares_actually_traded_abs - shares_to_sell_from_long
                                pos_data['quantity_shares'] -= remaining_shares_sold  # Становится отрицательным
                                pos_data[
                                    'average_entry_price_per_share'] = price_per_share  # Новая цена входа для шорта
                                enriched_trade_info["direction_str"] = "Продажа (закр.лонг, откр.шорт)"
                            print(
                                f"    Закрыто/уменьшено в лонге: {shares_to_sell_from_long} акций. P&L (без ком.): {realized_pnl_this_trade:.2f}")

                        else:  # Продажа для открытия/увеличения шорта
                            enriched_trade_info["direction_str"] = "Продажа (шорт)"
                            # prev_qty <= 0. abs(prev_qty) - кол-во акций в шорте до сделки
                            # initial_value_of_traded_shares - выручка от текущей продажи
                            new_total_proceeds_abs = (abs(prev_qty) * prev_avg_price) + initial_value_of_traded_shares
                            pos_data[
                                'quantity_shares'] -= shares_actually_traded_abs  # Увеличиваем абсолютное значение шорта
                            if abs(pos_data['quantity_shares']) > Decimal(0):
                                pos_data['average_entry_price_per_share'] = new_total_proceeds_abs / abs(
                                    pos_data['quantity_shares'])
                            else:  # Этого не должно быть, если мы открываем/увеличиваем шорт
                                pos_data['average_entry_price_per_share'] = Decimal(0)
                            print(f"    Открыт/увеличен шорт: {shares_actually_traded_abs} акций.")
                        realized_pnl_this_trade -= commission_rub  # Учитываем комиссию в P&L

                    portfolio_value_now_rub = current_cash_rub
                    if pos_data['quantity_shares'] > Decimal(0):  # Если лонг
                        portfolio_value_now_rub += pos_data[
                                                       'quantity_shares'] * current_market_price  # Используем текущую рыночную
                    elif pos_data['quantity_shares'] < Decimal(0):  # Если шорт
                        # Обязательства по шорту уменьшают стоимость портфеля
                        cost_of_short_position_at_market = abs(pos_data['quantity_shares']) * current_market_price
                        # Пример: кэш 10000, шорт 10 акций по 500 (рыночная). Обязательство 5000. Портфель = 10000 - 5000 = 5000.
                        # Если цена упала до 400, обязательство 4000. Портфель = 10000 - 4000 = 6000.
                        # Средняя цена продажи шорта была, например, 510.
                        # Кэш после продажи = кэш до + 10*510 - комиссия
                        # Для оценки портфеля, мы смотрим, сколько будет стоить откупить шорт СЕЙЧАС.
                        # Если мы "виртуально" откупаем шорт по текущей рыночной, то кэш изменится на (ср_цена_продажи - тек_цена_рынка) * кол-во
                        # Более простой способ: Стоимость портфеля = тек_кэш - стоимость_откупа_шорта_по_рынку
                        portfolio_value_now_rub = current_cash_rub - cost_of_short_position_at_market

                    portfolio_value_history.append((trade_action_details["timestamp"], portfolio_value_now_rub))
                    enriched_trade_info.update({
                        "shares_executed": int(shares_actually_traded_abs),
                        "executed_price_per_share_str": f"{price_per_share:.4f}",
                        "initial_cost_or_proceeds_rub_str": f"{initial_value_of_traded_shares:.2f}",
                        "commission_rub_str": f"{commission_rub:.2f}",
                        "realized_pnl_this_trade_str": f"{realized_pnl_this_trade:.2f}" if (
                                enriched_trade_info["direction_str"].startswith(
                                    "Покупка (закр") or  # P&L только при закрытии
                                enriched_trade_info["direction_str"].startswith(
                                    "Продажа (закр")) else f"({commission_rub:.2f})",  # Иначе только комиссия
                        "cash_after_trade_str": f"{current_cash_rub:.2f}",
                        "portfolio_value_after_trade_str": f"{portfolio_value_now_rub:.2f}",
                    })
                    # <ВАЛИДАЦИЯ 2.7: Расширенная информация о сделке и лог сделок>
                    print("  --- Расширенная информация о сделке (enriched_trade_info) ---")
                    if JUPYTER_DISPLAY_AVAILABLE:
                        display(pd.DataFrame([enriched_trade_info]))
                    else:
                        print(pd.DataFrame([enriched_trade_info]))

                    trade_log_enriched.append(enriched_trade_info)
                    print("  --- Лог сделок (trade_log_enriched, хвост) ---")
                    if JUPYTER_DISPLAY_AVAILABLE:
                        display(pd.DataFrame(trade_log_enriched).tail(3))
                    else:
                        print(pd.DataFrame(trade_log_enriched).tail(3))
                    # </ВАЛИДАЦИЯ 2.7>

                elif trade_action_details and not trade_action_details["success"]:
                    print(f"    Сделка не удалась: {trade_action_details.get('message', 'Нет деталей')}")

                print(
                    f"  Текущая позиция: {pos_data['quantity_shares']} акций, ср. цена входа/продажи: {pos_data['average_entry_price_per_share']:.4f}")
                print(f"  Текущий кэш: {current_cash_rub:.2f} RUB")

            elif overall_target_predicted_price is None:
                print("  Нет итогового прогноза цены. Торговые операции в этом цикле пропускаются.")
            elif current_market_price is None:
                print("  Нет текущей рыночной цены. Торговые операции в этом цикле пропускаются.")

            # <ВАЛИДАЦИЯ 2.8: Состояние портфеля и кэша в конце итерации>
            print(f"  --- Состояние в конце цикла {operations_count} ---")
            print(f"  Текущие позиции: {current_positions}")
            print(f"  Текущий кэш: {current_cash_rub:.2f} RUB")
            print("  История стоимости портфеля (хвост):")
            if portfolio_value_history:
                temp_portfolio_df = pd.DataFrame(portfolio_value_history,
                                                 columns=['timestamp_utc', 'portfolio_value_rub'])
                temp_portfolio_df['timestamp_msk'] = temp_portfolio_df['timestamp_utc'].apply(format_datetime_msk)
                if JUPYTER_DISPLAY_AVAILABLE:
                    display(temp_portfolio_df[['timestamp_msk', 'portfolio_value_rub']].tail(3))
                else:
                    print(temp_portfolio_df[['timestamp_msk', 'portfolio_value_rub']].tail(3))
            # </ВАЛИДАЦИЯ 2.8>

            if user_stop_requested:
                break

            print(
                f"\n  Ожидание {TRADE_INTERVAL_SECONDS // 60} мин перед следующим циклом и/или запросом на продолжение...")
            time.sleep(TRADE_INTERVAL_SECONDS)

            if INPUTIMEOUT_AVAILABLE:
                print("\n--- ЗАПРОС НА ПРОДОЛЖЕНИЕ ---")
                try:
                    prompt_message = (
                        f'Прошло {TRADE_INTERVAL_SECONDS // 60} мин. Продолжить торговлю? '
                        f'(да/нет/пусто=да, таймаут {USER_PROMPT_TIMEOUT_SECONDS} сек): '
                    )
                    user_response = inputimeout(prompt=prompt_message, timeout=USER_PROMPT_TIMEOUT_SECONDS)
                    user_response = user_response.strip().lower()

                    if user_response in ['нет', 'н', 'no', 'n']:
                        user_stop_requested = True
                        stop_loss_triggered_reason = "Пользователь запросил остановку."
                        print("Пользователь запросил остановку. Завершение работы...")
                    elif user_response in ['', 'да', 'д', 'yes', 'y']:
                        print("Ответ 'да' или пустой. Продолжаем торговлю...")
                    else:
                        print(f"Неизвестный ответ '{user_response}'. Продолжаем торговлю по умолчанию...")

                except TimeoutOccurred:
                    print(
                        f"Время на ответ ({USER_PROMPT_TIMEOUT_SECONDS} сек) истекло. Продолжаем торговлю по умолчанию...")
                except Exception as e_input:
                    print(f"Ошибка при запросе ввода у пользователя: {e_input}. Продолжаем торговлю по умолчанию.")
                    print("Это может произойти, если скрипт запущен в неинтерактивной среде.")
            else:
                print("Автоматическое продолжение (библиотека inputimeout недоступна).")

            if user_stop_requested:
                break

                # --- КОНЕЦ ОСНОВНОГО ЦИКЛА ---

        if stop_loss_triggered_reason:
            print(f"\n--- РОБОТ ОСТАНОВЛЕН. ПРИЧИНА: {stop_loss_triggered_reason} ---")
        elif user_stop_requested and not stop_loss_triggered_reason:
            print("\n--- Работа робота прервана пользователем. ---")
        elif not user_stop_requested and not stop_loss_triggered_reason:
            print("\n--- Завершение работы робота (естественное, например, закончились прогнозы). ---")

        final_market_price: Decimal | None = current_market_price
        if final_market_price is None and market_price_history:
            final_market_price = market_price_history[-1][1]

        if final_market_price is None:  # Попытка получить актуальную цену, если возможно
            final_market_price_check = get_current_price(client, INSTRUMENT_FIGI)
            if final_market_price_check is not None:
                final_market_price = final_market_price_check
                print(f"INFO: Получена актуальная финальная рыночная цена: {final_market_price:.4f}")
            else:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось получить актуальную финальную рыночную цену для {INSTRUMENT_FIGI}.")

        pos_data = current_positions[INSTRUMENT_FIGI]
        # Обработка случая, если final_market_price всё ещё None
        if final_market_price is None:
            if pos_data['quantity_shares'] != Decimal(0):  # Если есть открытые позиции
                if pos_data['average_entry_price_per_share'] != Decimal(0):  # Используем среднюю цену входа, если есть
                    final_market_price = pos_data['average_entry_price_per_share']
                    print(
                        f"ПРЕДУПРЕЖДЕНИЕ: Финальная рыночная цена не определена. Для расчета стоимости открытой позиции использована средняя цена входа/продажи ({final_market_price:.4f}). Это может быть неточно.")
                else:  # Крайний случай, если и средней цены нет (маловероятно при активной позиции)
                    final_market_price = Decimal("0.0001")  # Минимальное значение, чтобы избежать деления на ноль
                    print(
                        f"ПРЕДУПРЕЖДЕНИЕ: Финальная рыночная цена и средняя цена входа не определены. Использовано минимальное значение {final_market_price} для расчетов. Результаты могут быть неточными.")
            else:  # Нет открытых позиций
                final_market_price = Decimal("0.0001")  # Не влияет на расчеты, если позиций нет
                print("INFO: Финальная рыночная цена не определена, открытых позиций нет.")

        final_portfolio_value_rub = current_cash_rub

        if pos_data['quantity_shares'] > Decimal(0):
            value_of_long_position = pos_data['quantity_shares'] * final_market_price
            final_portfolio_value_rub = current_cash_rub + value_of_long_position
            print(f"INFO: На счете {pos_data['quantity_shares']} акций (лонг).")
            print(f"  Финальный кэш на счете: {current_cash_rub:.2f} RUB.")
            print(
                f"  Рыночная стоимость лонг-позиции ({pos_data['quantity_shares']} акций по {final_market_price:.4f} RUB/акция): {value_of_long_position:.2f} RUB.")

        elif pos_data['quantity_shares'] < Decimal(0):
            cost_to_cover_short_at_final_price = abs(pos_data['quantity_shares']) * final_market_price
            final_portfolio_value_rub = current_cash_rub - cost_to_cover_short_at_final_price
            print(f"INFO: На счете {pos_data['quantity_shares']} акций (шорт).")
            print(f"  Финальный кэш на счете: {current_cash_rub:.2f} RUB.")
            print(
                f"  Обязательства по откупу {abs(pos_data['quantity_shares'])} акций по финальной рыночной цене ({final_market_price:.4f} RUB/акция): {cost_to_cover_short_at_final_price:.2f} RUB.")
            unrealized_pnl_from_open_short = (pos_data['average_entry_price_per_share'] - final_market_price) * abs(
                pos_data['quantity_shares'])
            print(
                f"  Справочно: нереализованная прибыль/убыток по открытой шорт-позиции (ср. цена продажи {pos_data['average_entry_price_per_share']:.4f} RUB/акция): {unrealized_pnl_from_open_short:.2f} RUB.")
        else:
            print("INFO: Нет открытых позиций по акциям на конец торгов.")

        if not portfolio_value_history or portfolio_value_history[-1][1] != final_portfolio_value_rub:
            portfolio_value_history.append((datetime.now(timezone.utc), final_portfolio_value_rub))

        print(f"\n--- Итоговые результаты ---");
        print(f"Начальный баланс: {INITIAL_DEPOSIT_RUB:.2f} RUB");
        print(
            f"Финальный кэш (без учета стоимости/обязательств по открытым позициям на момент закрытия): {current_cash_rub:.2f} RUB")
        print(
            f"Кол-во акций в портфеле ({INSTRUMENT_FIGI}): {pos_data['quantity_shares']} (ср. цена входа/продажи: {pos_data['average_entry_price_per_share']:.4f})")
        print(
            f"Итоговая стоимость портфеля (кэш +/- стоимость/обязательства по позициям): {final_portfolio_value_rub:.2f} RUB")

        total_profit_rub = final_portfolio_value_rub - INITIAL_DEPOSIT_RUB
        total_profit_percent = (total_profit_rub / INITIAL_DEPOSIT_RUB) * Decimal(
            100) if INITIAL_DEPOSIT_RUB != Decimal(0) else Decimal(0)
        print(f"Общая прибыль/убыток: {total_profit_rub:.2f} RUB ({total_profit_percent:.2f}%)")

        print("\n--- Таблица сделок и P&L (Время MSK) ---")
        pnl_df = generate_pnl_table_df(trade_log_enriched, INSTRUMENT_FIGI)
        if not pnl_df.empty:
            # <ВАЛИДАЦИЯ 3.1: Итоговый DataFrame P&L (pnl_df)>
            print("Отображение pnl_df через display():")
            if JUPYTER_DISPLAY_AVAILABLE:
                display(pnl_df)
            else:
                print(pnl_df)  # Для случаев без IPython
            # </ВАЛИДАЦИЯ 3.1>

            # Опции для to_string, если он все же нужен для логов
            pd.set_option('display.max_rows', None);
            pd.set_option('display.max_columns', None);
            pd.set_option('display.width', 2000);
            pd.set_option('display.max_colwidth', None)
            # print(pnl_df.to_string(index=False)); # display() обычно предпочтительнее в Jupyter
            pnl_table_filename = os.path.join(OUTPUT_CSV_DIRECTORY, f"pnl_table_{INSTRUMENT_FIGI}_msk.csv")
            try:
                pnl_df.to_csv(pnl_table_filename, index=False, encoding='utf-8-sig');
                print(f"Таблица P&L сохранена: {pnl_table_filename}")
            except Exception as e_csv:
                print(f"Ошибка сохранения таблицы P&L в CSV: {e_csv}")

        else:
            print("Сделок не было, таблица P&L пуста.")

        plot_prices_and_trades(market_price_history, trade_log_enriched, predicted_price_history, INSTRUMENT_FIGI,
                               OUTPUT_CSV_DIRECTORY)
        plot_profitability(portfolio_value_history, INITIAL_DEPOSIT_RUB, INSTRUMENT_FIGI, OUTPUT_CSV_DIRECTORY)

        # --- Построение графика соотношения Прогноз/Реальная цена (всегда, если есть данные) ---
        if actual_prices_for_metrics_float and predictions_for_metrics_float and timestamps_for_metrics_utc and \
                len(timestamps_for_metrics_utc) >= MIN_SAMPLES_FOR_METRICS:
            print("\n--- Построение графика соотношения Прогноз/Реальная цена ---")
            # Конвертируем float обратно в Decimal для функции, как она ожидает
            actual_prices_decimal_for_ratio = [Decimal(str(p)) for p in actual_prices_for_metrics_float]
            predictions_decimal_for_ratio = [Decimal(str(p)) for p in predictions_for_metrics_float]
            plot_forecast_real_ratio(timestamps_for_metrics_utc,
                                     actual_prices_decimal_for_ratio,
                                     predictions_decimal_for_ratio,
                                     INSTRUMENT_FIGI,
                                     OUTPUT_CSV_DIRECTORY)
        else:
            print(
                "\n--- График соотношения Прогноз/Реальная цена не построен (недостаточно данных или данные отсутствуют). ---")

        try:
            client.sandbox.close_sandbox_account(account_id=sandbox_account_id);
            print(
                f"Счет песочницы {sandbox_account_id} закрыт.")
        except RequestError as e:
            print(f"Ошибка при закрытии счета {sandbox_account_id}: {e}")


if __name__ == "__main__":
    # Для работы в Jupyter Notebook, можно раскомментировать следующую строку,
    # чтобы графики отображались inline.
    # %matplotlib inline
    # Однако, если этот скрипт запускается и как обычный .py файл,
    # эта "магическая" команда вызовет ошибку.
    # plt.ion() # Интерактивный режим для .py, если нужно

    # Если вы хотите, чтобы настройки pandas для вывода (ширина и т.д.)
    # применялись глобально в Jupyter, их можно установить здесь:
    # pd.set_option('display.max_rows', 100)
    # pd.set_option('display.max_columns', 20)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.max_colwidth', 50)

    trading_bot_main()