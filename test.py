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
