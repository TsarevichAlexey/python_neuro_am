<<<<<<< HEAD
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import quotation_to_decimal
import os
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Frame, Scrollbar, Canvas, BOTH, Y, X, LEFT, RIGHT, TOP, BOTTOM, StringVar, ttk, PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import requests
from io import BytesIO
from PIL import Image, ImageTk  # Для работы с изображениями


# Укажите ваш API-токен
TOKEN = os.getenv("TINKOFF_TOKEN", "t.He-UEc5Mfe_E2w9CVSlZsCrqyoFacn52Yut_rwdsK8EclvfaxLlju33-26Z2cW22w0dDKNK0cNpxsyXLSGZv-g")
ACCOUNT_ID = '2039833462'

def get_portfolio_data():
    """Получает данные портфеля."""
    try:
        with Client(TOKEN) as client:
            portfolio = client.operations.get_portfolio(account_id=ACCOUNT_ID)
            positions_data = []
            for position in portfolio.positions:
                instrument = client.instruments.get_instrument_by(
                    id_type=1, id=position.figi
                ).instrument
                positions_data.append(
                    {
                        "name": instrument.name,
                        "ticker": instrument.ticker,
                        "figi": position.figi,
                        "type": instrument.instrument_type,
                        "quantity": quotation_to_decimal(position.quantity),
                        "average_price": quotation_to_decimal(
                            position.average_position_price
                        ),
                        "current_price": quotation_to_decimal(position.current_price),
                        "isin": instrument.isin, # Добавляем ISIN
                    }
                )
            return {
                "total_portfolio_value": quotation_to_decimal(
                    portfolio.total_amount_portfolio
                ),
                "expected_yield": quotation_to_decimal(portfolio.expected_yield),
                "positions": positions_data,
            }
    except Exception as e:
        print(f"❌ Ошибка при получении данных портфеля: {e}")
        return None


def get_stock_candles(figi, ticker):
    """Получает свечные данные за последний год."""
    try:
        with Client(TOKEN) as client:
            candles = client.market_data.get_candles(
                figi=figi,
                from_=datetime.now() - timedelta(days=365),
                to=datetime.now(),
                interval=CandleInterval.CANDLE_INTERVAL_DAY,
            ).candles
            prices = [quotation_to_decimal(candle.close) for candle in candles]
            dates = [candle.time for candle in candles]
            return dates, prices
    except Exception as e:
        print(f"❌ Ошибка при получении свечных данных для {ticker}: {e}")
        return None, None

def show_stock_chart(figi, ticker, position_data):
    """Отображает график."""
    dates, prices = get_stock_candles(figi, ticker)
    if dates and prices:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dates, prices)
        ax.set_title(f"График цен {ticker} (Московская биржа)")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Цена (RUB)")
        ax.grid(True)

        position_text = (
            f"Ваша позиция в {ticker}:\n"
            f"Количество: {int(position_data['quantity'])}\n"
            f"Средняя цена: {position_data['average_price']} RUB\n"
            f"Текущая стоимость: {position_data['current_price']} RUB"
        )
        ax.text(
            0.05,
            0.95,
            position_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        chart_window = Tk()
        chart_window.title(f"График и позиция {ticker}")
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, chart_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        chart_window.mainloop()
    else:
        print(f"Не удалось получить данные для графика {ticker}")
        error_label.config(text=f"Не удалось загрузить график для {ticker}")


def create_portfolio_pie_chart(portfolio_data, chart_frame):
    """Создает интерактивную диаграмму."""

    if not portfolio_data or not portfolio_data["positions"]:
        error_label.config(text="Не удалось загрузить данные портфеля.")
        return

    security_positions = [
        pos for pos in portfolio_data["positions"] if pos["type"] != "currency"
    ]
    labels = [pos["ticker"] for pos in security_positions]
    values = [
        float(pos["current_price"]) * float(pos["quantity"])
        for pos in security_positions
    ]

    if not labels or not values:
        error_label.config(text="Нет данных для диаграммы.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    num_wedges = len(values)
    base_colors = plt.cm.viridis([i / num_wedges for i in range(num_wedges)])
    darkened_colors = [mcolors.to_rgba(c, alpha=0.5) for c in base_colors]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=darkened_colors,
        wedgeprops=dict(width=0.7, edgecolor="w"),
        pctdistance=0.85,
    )
    ax.set_title("Состав вашего портфеля (без валюты)")
    ax.axis("equal")

    current_highlighted_wedge = None

    def highlight_wedge(wedge, color):
        wedge.set_color(color)
        wedge.set_alpha(1.0)
        canvas.draw_idle()

    def unhighlight_wedge(wedge, original_color):
        wedge.set_color(original_color)
        wedge.set_alpha(0.5)
        canvas.draw_idle()

    def on_motion(event):
        nonlocal current_highlighted_wedge

        if event.inaxes == ax:
            for i, wedge in enumerate(wedges):
                if wedge.contains_point((event.x, event.y)):
                    if current_highlighted_wedge != wedge:
                        if current_highlighted_wedge:
                            j = wedges.index(current_highlighted_wedge)
                            unhighlight_wedge(current_highlighted_wedge, darkened_colors[j])
                        highlight_wedge(wedge, base_colors[i])
                        current_highlighted_wedge = wedge
                    return

            if current_highlighted_wedge:
                j = wedges.index(current_highlighted_wedge)
                unhighlight_wedge(current_highlighted_wedge, darkened_colors[j])
                current_highlighted_wedge = None

    def on_click(event):
        if event.inaxes == ax:
            for i, wedge in enumerate(wedges):
                if wedge.contains_point((event.x, event.y)):
                    clicked_ticker = labels[i]
                    clicked_position = security_positions[i]
                    show_stock_chart(
                        clicked_position["figi"], clicked_ticker, clicked_position
                    )
                    break

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def get_company_logo(isin):
    """
    Получает URL логотипа компании по ISIN.
    Использует сервис Clearbit Logo API (требуется регистрация и API key).
    Если не удалось получить логотип, возвращает None.
    """
    #  Вам нужен API key от Clearbit
    clearbit_api_key = os.getenv("CLEARBIT_API_KEY", "your_clearbit_api_key")
    if not clearbit_api_key or clearbit_api_key == 'your_clearbit_api_key':
      print("Clearbit API key не установлен.  Установите CLEARBIT_API_KEY в переменных окружения.")
      return None

    try:
      # Ищем компанию по ISIN на другом сервисе, потому что Clearbit не всегда находит по ISIN
      # Можно использовать другой, если этот не работает
      search_url = f"https://www.openfigi.com/api/v1/mapping"
      headers = {'Content-Type': 'application/json'}
      data = [{"idType": "ID_ISIN", "idValue": isin, "exchCode": "RU"}]  # Указываем биржу (RU)
      response = requests.post(search_url, headers=headers, json=data)
      response.raise_for_status()  # Проверяем на ошибки HTTP
      search_results = response.json()


      if search_results and search_results[0].get('data'):
        domain = None
        # Пытаемся найти веб-сайт компании
        for item in search_results[0]['data']:
            # Можно добавить проверку на другие идентификаторы, если нужно
            if item.get('shareClassMIC') or item.get('compositeFIGI'):
              #  Попытка найти информацию о компании по FIGI через еще один запрос
              security_url = f"https://www.openfigi.com/api/v1/mapping"
              headers = {'Content-Type': 'application/json'}

              if item.get('compositeFIGI'):
                data = [{"idType": "ID_BB_GLOBAL", "idValue": item['compositeFIGI'], "exchCode": "RU"}]
              else:
                data = [{"idType": "ID_BB_GLOBAL", "idValue": item['shareClassFIGI'], "exchCode": "RU"}]

              response = requests.post(security_url, headers=headers, json=data)
              response.raise_for_status()
              security_info = response.json()
              if security_info and security_info[0].get('data') and security_info[0]['data'][0].get('securityType'): #Проверка существования ключа
                domain = security_info[0]['data'][0].get('securityType') # Получаем securityType
                # print(domain)
                break  # Выходим, если нашли сайт

        if domain:
          # Теперь используем Clearbit API для получения логотипа по домену
          clearbit_url = f"https://logo.clearbit.com/{domain}?size=64" # Указываем размер
          headers = {"Authorization": f"Bearer {clearbit_api_key}"}
          response = requests.get(clearbit_url, headers=headers, stream=True) # Потоковое получение
          response.raise_for_status()

          return response.raw # Возвращаем "сырой" (raw) объект ответа для обработки изображения

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении логотипа: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return None
    return None

def display_portfolio_positions(portfolio_data, positions_frame):
    """Отображает список позиций."""
    if not portfolio_data or not portfolio_data["positions"]:
        error_label.config(text="Не удалось загрузить позиции портфеля.")
        return

    for widget in positions_frame.winfo_children():
        widget.destroy()  # Очищаем фрейм

    security_positions = [
        pos for pos in portfolio_data["positions"] if pos["type"] != "currency"
    ]

    for position in security_positions:
        position_frame = Frame(positions_frame, bd=1, relief="groove", padx=5, pady=5)
        position_frame.pack(pady=2, fill=X)
        position_frame.bind("<Button-1>", lambda event, figi=position["figi"], ticker=position["ticker"], pos_data=position: show_stock_chart(figi, ticker, pos_data)) # Обработчик клика по всему фрейму

        # Логотип (с обработкой ошибок)
        logo_label = Label(position_frame) # Создаем label для лого
        logo_label.pack(side=LEFT)
        try:
            logo_data = get_company_logo(position['isin'])
            if logo_data:
                img = Image.open(logo_data)
                img = img.resize((32, 32), Image.Resampling.LANCZOS)  # Изменяем размер и сглаживаем
                photo = ImageTk.PhotoImage(img)
                logo_label.config(image=photo)
                logo_label.image = photo  # Сохраняем ссылку
            else:
                logo_label.config(text="Лого") # Текст если нет логотипа
        except Exception as e:
            print(f"Ошибка при загрузке/отображении лого: {e}")
            logo_label.config(text="Лого") #Текст, если ошибка



        profit_loss = (position["current_price"] - position["average_price"]) * position[
            "quantity"
        ]
        profit_loss_color = "green" if profit_loss >= 0 else "red"

        position_info_label = Label(
            position_frame,
            text=f"➡ {position['name']} ({position['ticker']})\n"
            f"  Тип: {position['type']}\n"
            f"  Количество: {int(position['quantity'])}\n"
            f"  Средняя цена: {position['average_price']} RUB\n"
            f"  Текущая стоимость: {position['current_price']} RUB",
            justify=LEFT,
        )
        position_info_label.pack(side=LEFT, padx=(5, 10)) # Добавил отступы

        profit_loss_label = Label(
            position_frame,
            text=f"Прибыль/Убыток: {profit_loss:.2f} RUB",
            fg=profit_loss_color,
        )
        profit_loss_label.pack(side=LEFT)
def display_as_list(portfolio_data, positions_frame):
    """Отображает портфель в виде списка."""

    # Очистка positions_frame
    for widget in positions_frame.winfo_children():
        widget.destroy()

    if not portfolio_data or not portfolio_data['positions']:
        error_label.config(text="Нет данных портфеля для отображения.")
        return

    # Создание прокручиваемой области
    canvas = Canvas(positions_frame)
    scrollbar = Scrollbar(positions_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Добавляем "прослушку" событий колеса мыши для прокрутки
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    display_portfolio_positions(portfolio_data, scrollable_frame)


def main_window():
    root = Tk()
    root.title("Портфель Тинькофф Инвестиции")

    global error_label
    error_label = Label(root, text="", fg="red")
    error_label.pack(pady=5)

    portfolio_data = get_portfolio_data()

    # Фреймы
    chart_frame = Frame(root)  # Для диаграммы
    positions_frame = Frame(root)  # Для списка позиций

    # Переменная для хранения текущего вида отображения
    display_mode = StringVar(value="list")  # "list" или "chart"

    # Функции для переключения вида
    def show_chart():
        positions_frame.pack_forget()  # Скрываем список
        chart_frame.pack(side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)
        create_portfolio_pie_chart(portfolio_data, chart_frame)

    def show_list():
        chart_frame.pack_forget()  # Скрываем диаграмму
        positions_frame.pack(side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)
        display_as_list(portfolio_data, positions_frame)

    # Кнопки переключения вида
    chart_button = Button(root, text="Диаграмма", command=show_chart)
    chart_button.pack(side=TOP, padx=5, pady=2)
    list_button = Button(root, text="Список", command=show_list)
    list_button.pack(side=TOP, padx=5, pady=2)

    # Изначально отображаем список
    show_list()

    root.mainloop()


if __name__ == "__main__":
    main_window()
=======
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
>>>>>>> 95657d1 (Первоначальная версия. Акция сбера, IBM. Разобраться, почему настолько точно! Провести более корректные тесты!)
