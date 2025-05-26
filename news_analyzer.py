# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import time  # For news processing (still relevant)
import requests  # For news processing

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from tinkoff.invest import Client, CandleInterval, RequestError
from tinkoff.invest.utils import now, quotation_to_decimal

try:
    import pandas_ta as ta
except ImportError:
    print("Библиотека pandas_ta не найдена. Пожалуйста, установите ее: pip install pandas_ta")
    # exit()

# For News Sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Предполагается, что config.py содержит:
# TOKEN = "ваш_тинькофф_токен"
# api_token = "ваш_newsapi_ключ" (ранее это был ключ Financemarker)
from config import api_token, TOKEN

import warnings

warnings.filterwarnings("ignore")

# %%
# --- ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ КОНФИГУРАЦИИ ---

# --- API Ключи ---
TINKOFF_API_TOKEN = TOKEN  # <--- ВАШ КЛЮЧ TINKOFF INVEST API
# Используем api_token из config.py как ключ для NewsAPI
NEWSAPI_KEY = api_token  # <--- ВАШ КЛЮЧ NEWSAPI
NEWSAPI_KEY_DEFAULT_PLACEHOLDER = "YOUR_NEWSAPI_KEY_PLACEHOLDER"  # Значение по умолчанию, если ключ не задан в config.py

# --- Параметры инструмента и общие для моделей ---
INSTRUMENT_FIGI = "BBG004730N88"  # SBER
FEATURE_COLUMN = 'close'  # Целевая колонка для прогнозирования цены
Z_SCORE_THRESHOLD = 3  # Порог для удаления выбросов по Z-оценке
N_CV_SPLITS = 4  # Количество фолдов в кросс-валидации
MAX_CV_WORKERS = min(N_CV_SPLITS, (os.cpu_count() or 1) * 2)  # Потоки для CV
TEST_SPLIT_SIZE = 0.2  # Доля данных для тестовой выборки
RANDOM_STATE = 42  # Для воспроизводимости

# --- Директория для сохранения CSV ---
OUTPUT_CSV_DIRECTORY = "future_price_predictions"

# --- Параметры Технических Индикаторов (TA) ---
SMA_PERIODS = [1, 5, 10]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD_DEV = 2
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH_K = 3

# --- Конфигурация для новостей (NewsAPI) ---
NEWSAPI_BASE_URL_EVERYTHING = "https://newsapi.org/v2/everything"
NEWS_API_PAGE_SIZE = 100  # NewsAPI позволяет до 100
NEWSAPI_HISTORY_LIMIT_DAYS = 28  # Ограничение бесплатного NewsAPI по глубине истории (примерно)
NEWSAPI_MAX_PAGES_PER_DAY = 5  # Ограничение на количество страниц при пагинации за день (100 * 5 = 500 новостей)

# Глобальные переменные для модели анализа тональности (News)
SENTIMENT_MODEL_NAME = "MonoHime/rubert-base-cased-sentiment-new"
sentiment_tokenizer = None
sentiment_model = None
sentiment_device = None
sentiment_model_loaded = False
sentiment_model_lock = Lock()

# --- Общий Lock для вывода в консоль ---
print_lock = Lock()

# --- Графики ---
MAX_GAP_MULTIPLIER_FOR_PLOT = 1.0  # Если разрыв > timeframe * multiplier, сжимаем

# Настройка GPU для TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        with print_lock:
            print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        with print_lock:
            print(f"Error setting memory growth: {e}. Must be set before GPU init.")


# %%
# --- БЛОК НОВОСТЕЙ: Функции для обработки новостей (адаптировано для NewsAPI) ---

def load_sentiment_model_if_needed(p_lock: Lock):
    global sentiment_tokenizer, sentiment_model, sentiment_device, sentiment_model_loaded
    if sentiment_model_loaded:
        return True
    with sentiment_model_lock:
        if sentiment_model_loaded:
            return True
        try:
            with p_lock:
                print(f"[NEWS_SENTIMENT_LOADER] Начало загрузки модели анализа тональности: {SENTIMENT_MODEL_NAME}...")
            sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
            sentiment_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sentiment_model.to(sentiment_device)
            sentiment_model.eval()
            sentiment_model_loaded = True
            with p_lock:
                print(
                    f"[NEWS_SENTIMENT_LOADER] Модель {SENTIMENT_MODEL_NAME} успешно загружена на устройство: {sentiment_device}.")
            return True
        except Exception as e:
            with p_lock:
                print(
                    f"[NEWS_SENTIMENT_LOADER] КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель анализа тональности {SENTIMENT_MODEL_NAME}. Ошибка: {e}")
            sentiment_model_loaded = False
            return False


def analyze_batch_sentiment(texts: list, p_lock: Lock):
    if not sentiment_model_loaded:
        if not load_sentiment_model_if_needed(p_lock):
            with p_lock:
                print("[NEWS_ANALYZER] Модель тональности не загружена. Анализ тональности невозможен.")
            return [0] * len(texts)

    inputs = sentiment_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(sentiment_device) for k, v in inputs.items()}
    sentiment_scores = []
    try:
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        for pred_idx in predictions:
            label = sentiment_model.config.id2label[pred_idx.item()]
            if label == 'positive':
                sentiment_scores.append(1)
            elif label == 'negative':
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
    except Exception as e:
        with p_lock:
            print(f"[NEWS_ANALYZER] Ошибка во время анализа тональности батча: {e}")
        return [0] * len(texts)
    return sentiment_scores


def get_search_query_for_instrument(instrument_identifier: str, p_lock: Lock) -> str:
    # Простой маппинг FIGI/тикера на поисковый запрос для NewsAPI
    # Это место можно улучшить, например, получая тикер или название компании через Tinkoff API
    if instrument_identifier == "BBG004730N88":  # SBER
        query = '"Сбербанк" OR "SBER"'
        with p_lock:
            print(f"[NEWS_QUERY_HELPER] Для FIGI {instrument_identifier} используется поисковый запрос: {query}")
        return query
    # Добавьте другие маппинги здесь:
    # if instrument_identifier == "BBG004730RP0":  # GAZP (пример)
    #    return '"Газпром" OR "GAZP"'

    # Если инструмент это тикер (например, "AAPL"), он может сработать напрямую
    # Если это неизвестный FIGI, прямой поиск по нему в NewsAPI скорее всего не даст результатов
    with p_lock:
        print(
            f"[NEWS_QUERY_HELPER] ВНИМАНИЕ: Для инструмента '{instrument_identifier}' не найдено специального маппинга. Используется как есть. Это может быть неэффективно для NewsAPI, если это FIGI.")
    return instrument_identifier


def fetch_real_news_from_newsapi(search_query: str,
                                 date_from_iso: str,  # YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                                 date_to_iso: str,  # YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                                 page_num: int,
                                 current_api_key: str,
                                 p_lock: Lock,
                                 model_name_prefix_for_log: str = ""):
    params = {
        'apiKey': current_api_key,
        'q': search_query,
        'from': date_from_iso,
        'to': date_to_iso,
        'language': 'ru',  # Можно сделать параметром
        'sortBy': 'publishedAt',  # или 'relevancy', 'popularity'
        'pageSize': NEWS_API_PAGE_SIZE,
        'page': page_num
    }

    log_prefix = f"[{model_name_prefix_for_log}]" if model_name_prefix_for_log else ""

    # Маскировка ключа в URL для логгирования
    masked_params_for_log = params.copy()
    if current_api_key and current_api_key != NEWSAPI_KEY_DEFAULT_PLACEHOLDER:
        masked_params_for_log['apiKey'] = "***API_KEY***"

    try:
        prepared_request = requests.Request('GET', NEWSAPI_BASE_URL_EVERYTHING, params=masked_params_for_log).prepare()
        full_url_for_log = prepared_request.url
    except Exception as e_url:
        full_url_for_log = f"{NEWSAPI_BASE_URL_EVERYTHING}?params=... (ошибка сборки URL для лога: {e_url})"

    with p_lock:
        print(f"{log_prefix}[NEWS_API_REQUEST] Запрос (страница {page_num}) на URL: {full_url_for_log}")

    if not current_api_key or current_api_key == NEWSAPI_KEY_DEFAULT_PLACEHOLDER:
        with p_lock:
            print(
                f"{log_prefix}[NEWS_STUB] ЗАГЛУШКА: API ключ NewsAPI не настроен или является плейсхолдером. Мок-ответ для '{search_query}', {date_from_iso} - {date_to_iso}, стр. {page_num}.")
        mock_news_articles = []
        num_mock_news = np.random.randint(0, min(5, NEWS_API_PAGE_SIZE // 2))  # Меньше новостей в заглушке
        try:
            base_ts = pd.Timestamp(date_from_iso.split('T')[0])  # Берем только дату
            for i in range(num_mock_news):
                h, m, s = np.random.randint(0, 24), np.random.randint(0, 60), np.random.randint(0, 60)
                news_ts = base_ts.replace(hour=h, minute=m, second=s)
                sentiment_type = np.random.choice(["positive", "negative", "neutral"])
                mock_news_articles.append({
                    "publishedAt": news_ts.isoformat(timespec='seconds') + "Z",  # NewsAPI формат
                    "title": f"Mock NewsAPI Title {i + 1} ({sentiment_type}) for {search_query}",
                    "description": f"Mock NewsAPI description for {search_query} ({sentiment_type}). " + "word " * 15,
                    "content": f"Mock NewsAPI content for {search_query} ({sentiment_type}). " + "word " * 10 + "[+50 chars]",
                    "source": {"name": "MockSource"}
                })
        except Exception as e_mock:
            with p_lock:
                print(f"{log_prefix}[NEWS_STUB] Ошибка генерации мок-данных NewsAPI: {e_mock}")
        return mock_news_articles, len(mock_news_articles)  # (articles, totalResults)

    try:
        response = requests.get(NEWSAPI_BASE_URL_EVERYTHING, params=params, timeout=45)
        response.raise_for_status()  # Проверка на HTTP ошибки (4xx, 5xx)
        data = response.json()

        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            total_results = data.get('totalResults', 0)
            return articles, total_results
        else:
            api_error_code = data.get('code', 'UnknownCode')
            api_error_message = data.get('message', 'Unknown error from NewsAPI')
            with p_lock:
                print(
                    f"{log_prefix}[NEWS_API_ERROR] Ошибка от NewsAPI для '{search_query}' ({date_from_iso}-{date_to_iso}): Code: {api_error_code}, Message: {api_error_message}")
            return [], 0

    except requests.exceptions.HTTPError as e:
        with p_lock:
            print(
                f"{log_prefix}[NEWS_API_ERROR] HTTP ошибка для '{search_query}' ({date_from_iso}-{date_to_iso}): {e}. Ответ: {e.response.text[:200] if e.response else 'N/A'}")
    except requests.exceptions.RequestException as e:
        with p_lock:
            print(
                f"{log_prefix}[NEWS_API_ERROR] Сетевая ошибка для '{search_query}' ({date_from_iso}-{date_to_iso}): {e}")
    except ValueError as e:  # JSONDecodeError
        with p_lock:
            print(
                f"{log_prefix}[NEWS_API_ERROR] Ошибка декодирования JSON для '{search_query}' ({date_from_iso}-{date_to_iso}): {e}")
    except Exception as e:
        with p_lock:
            print(
                f"{log_prefix}[NEWS_API_ERROR] Неизвестная ошибка для '{search_query}' ({date_from_iso}-{date_to_iso}): {e}")
    return [], 0  # (articles, totalResults)


def fetch_day_news_with_pagination(search_query: str, target_date_pd: pd.Timestamp, current_api_key: str,
                                   p_lock: Lock, model_name_prefix: str):
    # NewsAPI `from` and `to` parameters for the /everything endpoint should be in ISO 8601 format.
    # We'll fetch news for the entirety of the target_date_pd.
    date_from_api_str = target_date_pd.strftime('%Y-%m-%dT00:00:00')
    date_to_api_str = target_date_pd.strftime('%Y-%m-%dT23:59:59')

    all_articles_for_day = []
    current_page = 1
    total_results_for_query = 0  # Будет обновлено из первого ответа API

    log_prefix_day_fetch = f"[{model_name_prefix}][NEWS_DAY_FETCHER:{target_date_pd.strftime('%Y-%m-%d')}]"

    while current_page <= NEWSAPI_MAX_PAGES_PER_DAY:
        articles_page, total_results_on_page = fetch_real_news_from_newsapi(
            search_query, date_from_api_str, date_to_api_str, current_page,
            current_api_key, p_lock, model_name_prefix
        )

        if current_page == 1:  # Обновляем общее количество результатов при первом запросе
            total_results_for_query = total_results_on_page
            if total_results_for_query == 0 and not articles_page:  # Если первый запрос пуст и totalResults=0
                with p_lock: print(f"{log_prefix_day_fetch} Новостей не найдено (totalResults=0).")
                break

        if not articles_page:  # Если страница пуста (не из-за totalResults=0, а, например, ошибка или конец данных)
            with p_lock: print(
                f"{log_prefix_day_fetch} Страница {current_page} не вернула статей. Завершение пагинации.")
            break

        all_articles_for_day.extend(articles_page)

        # Проверяем, нужно ли продолжать пагинацию
        # len(all_articles_for_day) >= total_results_for_query - это условие для остановки, если мы получили все.
        # len(articles_page) < NEWS_API_PAGE_SIZE - если последняя страница была неполной.
        if len(all_articles_for_day) >= total_results_for_query or len(articles_page) < NEWS_API_PAGE_SIZE:
            with p_lock: print(
                f"{log_prefix_day_fetch} Пагинация завершена на стр. {current_page}. Собрано {len(all_articles_for_day)}/{total_results_for_query} статей.")
            break

        current_page += 1
        if current_page > NEWSAPI_MAX_PAGES_PER_DAY:
            with p_lock: print(
                f"{log_prefix_day_fetch} Достигнут лимит страниц ({NEWSAPI_MAX_PAGES_PER_DAY}) для пагинации. Собрано {len(all_articles_for_day)} статей.")

    return all_articles_for_day


def process_instrument_news(instrument_identifier: str,  # Был figi, теперь общий идентификатор
                            news_fetch_days_param: int,
                            candle_interval_timedelta: timedelta,
                            p_lock: Lock,
                            model_name_prefix: str = ""):
    output_columns = ['news_sentiment_mean', 'news_sentiment_sum', 'news_count',
                      'news_positive_count', 'news_negative_count', 'news_neutral_count']
    empty_df_result = pd.DataFrame(columns=output_columns).set_index(pd.DatetimeIndex([]))
    for col in output_columns:
        dtype = int if 'count' in col else float
        empty_df_result[col] = pd.Series(dtype=dtype)

    if not load_sentiment_model_if_needed(p_lock): return empty_df_result

    search_query = get_search_query_for_instrument(instrument_identifier, p_lock)

    # Учет ограничения NewsAPI по истории (NEWSAPI_HISTORY_LIMIT_DAYS)
    end_datetime_utc_naive = (pd.Timestamp.now(tz=timezone.utc).normalize() + timedelta(days=1) - timedelta(
        microseconds=1)).tz_localize(None)

    requested_start_date_naive = (
                pd.Timestamp.now(tz=timezone.utc).normalize() - timedelta(days=news_fetch_days_param)).tz_localize(None)
    earliest_possible_start_date_newsapi_naive = (
                pd.Timestamp.now(tz=timezone.utc).normalize() - timedelta(days=NEWSAPI_HISTORY_LIMIT_DAYS)).tz_localize(
        None)

    start_datetime_utc_naive = max(requested_start_date_naive, earliest_possible_start_date_newsapi_naive)

    log_prefix_news = f"[{model_name_prefix}][NEWS_PROCESSOR]"
    with p_lock:
        print(
            f"{log_prefix_news} Запуск обработки новостей для запроса: '{search_query}' (исходный идентификатор: {instrument_identifier}).")
        if requested_start_date_naive < earliest_possible_start_date_newsapi_naive:
            print(
                f"{log_prefix_news} ВНИМАНИЕ: Запрошенная история новостей ({news_fetch_days_param} дней) превышает лимит NewsAPI ({NEWSAPI_HISTORY_LIMIT_DAYS} дней для бесплатного тарифа). Новости будут запрошены начиная с {start_datetime_utc_naive.strftime('%Y-%m-%d')}.")
        print(
            f"{log_prefix_news} Запрошенный период для новостей (UTC naive): С: {start_datetime_utc_naive.strftime('%Y-%m-%d %H:%M:%S')} ПО: {end_datetime_utc_naive.strftime('%Y-%m-%d %H:%M:%S')}")

    processed_raw_news = []
    current_api_request_date = start_datetime_utc_naive.normalize()
    loop_end_api_request_date = end_datetime_utc_naive.normalize()

    dates_to_fetch_for_api = []
    while current_api_request_date <= loop_end_api_request_date:
        dates_to_fetch_for_api.append(current_api_request_date)
        current_api_request_date += timedelta(days=1)

    if not dates_to_fetch_for_api:
        with p_lock: print(f"{log_prefix_news} Нет дат для запроса новостей.")
        return empty_df_result

    max_workers_news_fetch = min(8, len(dates_to_fetch_for_api), (os.cpu_count() or 1) * 2)

    with ThreadPoolExecutor(max_workers=max_workers_news_fetch,
                            thread_name_prefix=f"{model_name_prefix}_NewsFetch") as executor:
        future_to_date = {
            executor.submit(fetch_day_news_with_pagination, search_query, fetch_date, NEWSAPI_KEY, p_lock,
                            model_name_prefix): fetch_date
            for fetch_date in dates_to_fetch_for_api
        }
        for future in as_completed(future_to_date):
            fetch_date = future_to_date[future]
            try:
                daily_news_items = future.result()  # Это список статей от NewsAPI
                if daily_news_items:
                    for news_item in daily_news_items:
                        try:
                            title = news_item.get('title', '')
                            description = news_item.get('description', '')

                            # Используем title + description как основной текст. Content часто урезан.
                            text_content = f"{title} {description}".strip()
                            if not text_content:  # Если оба пусты, пробуем content (хотя он может быть неполон)
                                text_content = news_item.get('content', '')
                                if text_content and "[+" in text_content and text_content.endswith(
                                        "chars]"):  # Удаляем индикатор неполноты, если есть
                                    text_content = text_content[:text_content.rfind("[+")]
                                text_content = text_content.strip()

                            if not text_content: continue  # Пропускаем новость без текста

                            ts_str = news_item.get('publishedAt')  # NewsAPI поле
                            if not ts_str: continue

                            timestamp = pd.to_datetime(ts_str, errors='coerce')
                            if pd.isna(timestamp):
                                with p_lock: print(
                                    f"{log_prefix_news} Не удалось распознать дату: {ts_str} из новости: {str(news_item)[:100]}")
                                continue

                            # Приводим к UTC naive
                            if timestamp.tzinfo is not None:
                                timestamp = timestamp.tz_convert('UTC').tz_localize(None)
                            else:
                                timestamp = timestamp.tz_localize('UTC').tz_localize(
                                    None)  # Если была наивная, считаем UTC

                            if not (start_datetime_utc_naive <= timestamp <= end_datetime_utc_naive):
                                continue  # Новость не попадает в наш целевой диапазон дат

                            processed_raw_news.append({'time': timestamp, 'text': text_content})
                        except Exception as e_item:
                            with p_lock:
                                print(
                                    f"{log_prefix_news} Ошибка парсинга полей новости NewsAPI: {str(news_item)[:200]}. Детали: {e_item}")
            except Exception as e_future:
                with p_lock:
                    print(
                        f"{log_prefix_news} Ошибка получения новостей за {fetch_date.strftime('%Y-%m-%d')}: {e_future}")

    if not processed_raw_news:
        with p_lock: print(
            f"{log_prefix_news} Новости для запроса '{search_query}' не найдены или не прошли обработку.")
        return empty_df_result

    with p_lock:
        print(
            f"{log_prefix_news} Получено {len(processed_raw_news)} новостей для анализа тональности (запрос: '{search_query}').")
    processed_raw_news.sort(key=lambda x: x['time'])
    news_texts_for_analysis = [item['text'] for item in processed_raw_news]
    news_timestamps_for_analysis = [item['time'] for item in processed_raw_news]

    sentiment_batch_size = 32
    all_sentiment_scores = []
    for i in range(0, len(news_texts_for_analysis), sentiment_batch_size):
        batch_texts = news_texts_for_analysis[i:i + sentiment_batch_size]
        if not batch_texts: continue
        all_sentiment_scores.extend(analyze_batch_sentiment(batch_texts, p_lock))

    analyzed_news_data = []
    if len(all_sentiment_scores) == len(news_timestamps_for_analysis):
        for i in range(len(all_sentiment_scores)):
            analyzed_news_data.append(
                {'time': news_timestamps_for_analysis[i], 'sentiment_score': all_sentiment_scores[i]})
    else:
        with p_lock:
            print(
                f"{log_prefix_news} ОШИБКА: несоответствие длин времен ({len(news_timestamps_for_analysis)}) и оценок ({len(all_sentiment_scores)}).")
        return empty_df_result

    if not analyzed_news_data:
        with p_lock: print(f"{log_prefix_news} Нет данных после анализа тональности.")
        return empty_df_result

    news_df = pd.DataFrame(analyzed_news_data)
    if news_df.empty:
        with p_lock: print(f"{log_prefix_news} DataFrame новостей пуст после анализа.")
        return empty_df_result
    news_df.set_index('time', inplace=True)

    total_seconds = candle_interval_timedelta.total_seconds()
    if total_seconds <= 0:
        with p_lock: print(f"{log_prefix_news} Ошибка: интервал свечей <= 0 ({total_seconds}s).")
        return empty_df_result

    resample_rule_parts = []
    days, rem_sec = divmod(total_seconds, 86400)
    hours, rem_sec = divmod(rem_sec, 3600)
    mins, secs = divmod(rem_sec, 60)

    if days > 0: resample_rule_parts.append(f"{int(days)}D")
    if hours > 0: resample_rule_parts.append(f"{int(hours)}H")
    if mins > 0: resample_rule_parts.append(f"{int(mins)}min")
    if secs > 0: resample_rule_parts.append(f"{int(secs)}S")

    if not resample_rule_parts:
        if total_seconds > 0:
            resample_rule = f"{int(total_seconds)}S"
        else:
            with p_lock:
                print(f"{log_prefix_news} Не удалось определить правило resample для {candle_interval_timedelta}.")
            return empty_df_result
    else:
        resample_rule = "".join(resample_rule_parts)

    try:
        aggregated_news_df = news_df['sentiment_score'].resample(rule=resample_rule, closed='left', label='left').agg(
            news_sentiment_mean='mean', news_sentiment_sum='sum', news_count='count',
            news_positive_count=lambda x: (x > 0).sum(), news_negative_count=lambda x: (x < 0).sum(),
            news_neutral_count=lambda x: (x == 0).sum()
        )
    except Exception as e_resample:
        with p_lock:
            print(f"{log_prefix_news} Ошибка ресемплинга новостей по правилу '{resample_rule}': {e_resample}")
        return empty_df_result

    if 'news_sentiment_mean' in aggregated_news_df.columns:
        aggregated_news_df['news_sentiment_mean'] = aggregated_news_df['news_sentiment_mean'].fillna(0)
    aggregated_news_df = aggregated_news_df.fillna(0)

    for col in output_columns:
        if col not in aggregated_news_df: aggregated_news_df[col] = 0
        if 'count' in col:
            aggregated_news_df[col] = aggregated_news_df[col].astype(int)
        elif 'mean' in col or 'sum' in col:
            aggregated_news_df[col] = aggregated_news_df[col].astype(float)

    with p_lock:
        print(
            f"{log_prefix_news} Агрегировано {len(aggregated_news_df)} записей новостных признаков (интервал: {resample_rule}).")
    return aggregated_news_df[output_columns]


# %%
# --- БЛОК 0: Вспомогательные функции (Общие и для Tinkoff) ---

def save_predictions_to_csv(
        future_predictions_df: pd.DataFrame,
        model_name: str,
        instrument_figi: str,
        target_feature_name: str,
        candle_interval_name: str,
        output_dir: str = OUTPUT_CSV_DIRECTORY,
        p_lock: Lock = print_lock
):
    log_prefix_csv = f"[{model_name}][CSV_SAVE]"
    if future_predictions_df is None or future_predictions_df.empty:
        with p_lock:
            print(f"{log_prefix_csv} DataFrame с прогнозами пуст. CSV-файл не будет создан.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{instrument_figi}_{model_name}_{timestamp_str}.csv"
        filepath = os.path.join(output_dir, filename)
        df_to_save = future_predictions_df.copy()
        df_to_save.reset_index(inplace=True)
        df_to_save['model_name'] = model_name
        df_to_save['instrument_figi'] = instrument_figi
        df_to_save['target_feature'] = target_feature_name
        df_to_save['candle_interval'] = candle_interval_name
        df_to_save['prediction_made_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        df_to_save.rename(columns={'datetime': 'future_timestamp_utc',
                                   'predicted_value': f'predicted_{target_feature_name}'},
                          inplace=True)
        ordered_columns = [
            'model_name', 'instrument_figi', 'target_feature', 'candle_interval',
            'prediction_made_at_utc', 'future_timestamp_utc', f'predicted_{target_feature_name}'
        ]
        df_to_save = df_to_save.reindex(columns=ordered_columns)
        df_to_save.to_csv(filepath, index=False, encoding='utf-8')
        with p_lock:
            print(f"{log_prefix_csv} Прогнозы успешно сохранены в файл: {filepath}")
    except Exception as e:
        with p_lock:
            print(f"{log_prefix_csv} ОШИБКА при сохранении прогнозов в CSV: {e}")
            import traceback
            print(traceback.format_exc())


def get_timedelta_from_candle_interval(interval: CandleInterval) -> timedelta:
    mapping = {
        CandleInterval.CANDLE_INTERVAL_1_MIN: timedelta(minutes=1),
        CandleInterval.CANDLE_INTERVAL_5_MIN: timedelta(minutes=5),
        CandleInterval.CANDLE_INTERVAL_10_MIN: timedelta(minutes=10),
        CandleInterval.CANDLE_INTERVAL_15_MIN: timedelta(minutes=15),
        CandleInterval.CANDLE_INTERVAL_30_MIN: timedelta(minutes=30),
        CandleInterval.CANDLE_INTERVAL_HOUR: timedelta(hours=1),
        CandleInterval.CANDLE_INTERVAL_2_HOUR: timedelta(hours=2),
        CandleInterval.CANDLE_INTERVAL_4_HOUR: timedelta(hours=4),
        CandleInterval.CANDLE_INTERVAL_DAY: timedelta(days=1),
        CandleInterval.CANDLE_INTERVAL_WEEK: timedelta(weeks=1),
        CandleInterval.CANDLE_INTERVAL_MONTH: timedelta(days=30)
    }
    if interval in mapping:
        return mapping[interval]
    else:
        with print_lock:
            print(f"Предупреждение: Неизвестный интервал свечи {interval}, используется 1 минута по умолчанию.")
        return timedelta(minutes=1)


def generate_plotted_times_and_values(real_timestamps, values, timeframe_timedelta_expected,
                                      max_gap_multiplier=MAX_GAP_MULTIPLIER_FOR_PLOT):
    if not hasattr(real_timestamps, '__len__') or len(real_timestamps) == 0:
        return np.array([]), np.array([])
    if not hasattr(values, '__len__') or len(values) == 0:
        return np.array([]), np.array([])
    if len(real_timestamps) != len(values):
        with print_lock:
            print(
                f"ERROR in generate_plotted_times_and_values: Length mismatch ({len(real_timestamps)} vs {len(values)}). Returning empty.")
        return np.array([]), np.array([])

    real_timestamps_np = pd.to_datetime(real_timestamps).to_numpy()
    values_np = np.asarray(values)

    plotted_ts_list = [pd.Timestamp(real_timestamps_np[0])]
    for i in range(1, len(real_timestamps_np)):
        current_real_ts_pd = pd.Timestamp(real_timestamps_np[i])
        prev_real_ts_pd = pd.Timestamp(real_timestamps_np[i - 1])
        last_plotted_ts_pd = plotted_ts_list[-1]
        actual_diff = current_real_ts_pd - prev_real_ts_pd
        if actual_diff > timeframe_timedelta_expected * max_gap_multiplier:
            plotted_ts_list.append(last_plotted_ts_pd + timeframe_timedelta_expected)
        else:
            plotted_ts_list.append(last_plotted_ts_pd + actual_diff)
    return np.array(plotted_ts_list), values_np


# %%
# --- БЛОК 1: Загрузка и первичная обработка данных (Tinkoff Свечи и TA) ---
def get_tinkoff_candles(token, figi, days, interval, model_name=""):
    log_prefix_candles = f"[{model_name}][CANDLES_TINKOFF]"
    with print_lock:
        print(f"{log_prefix_candles} Загрузка данных: {days} дней, интервал {interval.name}")

    def fetch_day_data(day_offset):
        # Убедимся что from_ и to_ - это datetime объекты, а не Timestamp
        # SDK может быть чувствителен к этому. now() из SDK возвращает datetime.
        # pd.Timestamp.now() - pd.Timestamp.
        # Используем now() из SDK для согласованности.
        end_dt_sdk = now() - timedelta(days=day_offset)
        start_dt_sdk = end_dt_sdk - timedelta(days=1)

        try:
            with Client(token) as client:  # Клиент лучше создавать внутри потока
                candles = list(client.get_all_candles(
                    figi=figi,
                    from_=start_dt_sdk,
                    to=end_dt_sdk,
                    interval=interval
                ))
                return candles
        except RequestError as e:
            with print_lock:
                print(f"{log_prefix_candles} Ошибка при загрузке данных за {start_dt_sdk.date()}: {e}")
            return []
        except Exception as e_generic:
            with print_lock:
                print(
                    f"{log_prefix_candles} Неизвестная ошибка при загрузке данных за {start_dt_sdk.date()}: {e_generic}")
            return []

    max_fetch_workers = min(10, days if days > 0 else 1)
    all_candles_nested = []
    with ThreadPoolExecutor(max_workers=max_fetch_workers, thread_name_prefix=f"{model_name}_CandleFetch") as executor:
        futures = [executor.submit(fetch_day_data, day_offset) for day_offset in range(days)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_candles_nested.append(result)

    all_candles = [candle for sublist in all_candles_nested for candle in sublist]

    if not all_candles:
        with print_lock:
            print(f"{log_prefix_candles} Не удалось загрузить свечи.")
        return pd.DataFrame()

    df = pd.DataFrame([{
        'time': pd.Timestamp(c.time).tz_localize(None),
        'open': float(quotation_to_decimal(c.open)),
        'high': float(quotation_to_decimal(c.high)),
        'low': float(quotation_to_decimal(c.low)),
        'close': float(quotation_to_decimal(c.close)),
        'volume': c.volume
    } for c in all_candles])

    if df.empty:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=['time'])
    df = df.set_index('time').sort_index()
    df.dropna(inplace=True)

    with print_lock:
        print(f"{log_prefix_candles} Загружено всего {len(df)} уникальных свечей.")
        print(f"{log_prefix_candles} Размер DataFrame после загрузки и базовой очистки: {df.shape}")
    return df


def add_technical_indicators(df: pd.DataFrame, model_name="") -> pd.DataFrame:
    log_prefix_ta = f"[{model_name}][TECH_INDICATORS]"
    is_future_ta_calc = "-future_TA" in model_name

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            with print_lock:
                print(
                    f"{log_prefix_ta} Ошибка: Индекс DataFrame не является DatetimeIndex и не может быть преобразован: {e}")
            return df.copy()

    if not is_future_ta_calc:
        with print_lock:
            print(f"{log_prefix_ta} Добавление технических индикаторов... Исходный размер df: {df.shape}")

    df_ta = df.copy()
    rename_map = {}
    required_cols = ['open', 'high', 'low', 'close']
    for col_standard in required_cols + ['volume']:
        if col_standard not in df_ta.columns and col_standard.capitalize() in df_ta.columns:
            rename_map[col_standard.capitalize()] = col_standard
    if rename_map:
        df_ta.rename(columns=rename_map, inplace=True)
    df_ta.columns = df_ta.columns.str.lower()

    missing_cols = [col for col in required_cols if col not in df_ta.columns]
    if missing_cols:
        with print_lock:
            print(
                f"{log_prefix_ta} Предупреждение: Отсутствуют необходимые колонки для TA: {', '.join(missing_cols)}. Некоторые индикаторы могут быть не рассчитаны.")

    if 'close' in df_ta.columns:
        for period in SMA_PERIODS:
            df_ta[f'sma_{period}'] = ta.sma(df_ta['close'], length=period)
        macd = ta.macd(df_ta['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None and not macd.empty: df_ta = pd.concat([df_ta, macd], axis=1)
        bbands = ta.bbands(df_ta['close'], length=BBANDS_PERIOD, std=BBANDS_STD_DEV)
        if bbands is not None and not bbands.empty: df_ta = pd.concat([df_ta, bbands], axis=1)
        df_ta['rsi'] = ta.rsi(df_ta['close'], length=RSI_PERIOD)

    if all(col in df_ta.columns for col in ['high', 'low', 'close']):
        df_ta['atr'] = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=ATR_PERIOD)
        stoch = ta.stoch(df_ta['high'], df_ta['low'], df_ta['close'], k=STOCH_K, d=STOCH_D, smooth_k=STOCH_SMOOTH_K)
        if stoch is not None and not stoch.empty: df_ta = pd.concat([df_ta, stoch], axis=1)

    if all(col in df_ta.columns for col in ['close', 'volume']):
        df_ta['obv'] = ta.obv(df_ta['close'], df_ta['volume'])
    elif not is_future_ta_calc:
        with print_lock:
            print(f"{log_prefix_ta} Предупреждение: Колонка 'volume' или 'close' отсутствует, OBV не рассчитан.")

    if all(col in df_ta.columns for col in ['high', 'low', 'close', 'volume']):
        df_ta['typical_price_vol'] = ((df_ta['high'] + df_ta['low'] + df_ta['close']) / 3) * df_ta['volume']
        df_ta['cum_vol'] = df_ta['volume'].cumsum()
        df_ta['cum_typical_price_vol'] = df_ta['typical_price_vol'].cumsum()
        df_ta['vwap'] = np.where(df_ta['cum_vol'] != 0, df_ta['cum_typical_price_vol'] / df_ta['cum_vol'], np.nan)
        df_ta.drop(columns=['typical_price_vol', 'cum_vol', 'cum_typical_price_vol'], inplace=True, errors='ignore')
    elif not is_future_ta_calc:
        with print_lock:
            print(
                f"{log_prefix_ta} Предупреждение: Не удалось рассчитать VWAP, отсутствуют необходимые колонки (high, low, close, volume).")

    df_ta.replace([np.inf, -np.inf], np.nan, inplace=True)

    if not is_future_ta_calc:
        initial_nans = df_ta.isnull().sum().max()
        with print_lock:
            print(
                f"{log_prefix_ta} Технические индикаторы добавлены. NaN в начале (из-за периодов TA): ~{initial_nans}")
            print(f"{log_prefix_ta} Размер df после ТА (до dropna): {df_ta.shape}")
    return df_ta


def remove_outliers_zscore(df, column_name, threshold=3, model_name=""):
    log_prefix_outliers = f"[{model_name}][OUTLIERS]"
    if column_name not in df.columns:
        with print_lock: print(f"{log_prefix_outliers} Колонка '{column_name}' не найдена для удаления выбросов.")
        return df.copy()

    series = df[column_name].astype(float)
    mean = series.mean()
    std = series.std()

    if std == 0 or pd.isna(std) or std < 1e-9:
        with print_lock: print(
            f"{log_prefix_outliers} Стандартное отклонение для '{column_name}' близко к нулю или NaN. Выбросы не удаляются.")
        return df.copy()

    z_scores = (series - mean) / std
    outliers_mask = np.abs(z_scores) > threshold
    num_outliers = outliers_mask.sum()
    df_cleaned = df.copy()

    if num_outliers > 0:
        with print_lock:
            print(
                f"{log_prefix_outliers} Найдено и будет удалено {num_outliers} выбросов в '{column_name}' (Z-score > {threshold}).")
        df_cleaned = df[~outliers_mask]
    return df_cleaned


def plot_data_with_outliers(original_df, cleaned_df, column_name, threshold, model_name="", timeframe_td=None):
    log_prefix_plot = f"[{model_name}][PLOT_OUTLIERS]"
    if original_df.empty or column_name not in original_df.columns:
        with print_lock: print(
            f"{log_prefix_plot} Невозможно построить график выбросов: DataFrame пуст или нет колонки '{column_name}'.")
        return

    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(2, 1, 1)
    if timeframe_td is not None:
        plotted_ts_orig, mapped_values_orig = generate_plotted_times_and_values(
            original_df.index, original_df[column_name], timeframe_td
        )
        if len(plotted_ts_orig) > 0:
            ax1.plot(plotted_ts_orig, mapped_values_orig, label=f'Исходные ({column_name})')
            mean_orig = original_df[column_name].mean()
            std_orig = original_df[column_name].std()
            if std_orig > 0 and not pd.isna(std_orig):
                outliers_mask_bool = np.abs((original_df[column_name] - mean_orig) / std_orig) > threshold
                outliers_orig_df = original_df[outliers_mask_bool]
                if not outliers_orig_df.empty:
                    map_real_to_plotted = pd.Series(data=plotted_ts_orig, index=original_df.index)
                    valid_outlier_indices = outliers_orig_df.index.intersection(map_real_to_plotted.index)
                    if not valid_outlier_indices.empty:
                        outliers_plotted_times = map_real_to_plotted.loc[valid_outlier_indices]
                        ax1.scatter(outliers_plotted_times, outliers_orig_df.loc[valid_outlier_indices, column_name],
                                    color='red', label=f'Выбросы (Z > {threshold})', s=50, zorder=5)
        else:
            ax1.plot(original_df.index, original_df[column_name], label=f'Исходные ({column_name}) (fallback plot)')
    else:
        ax1.plot(original_df.index, original_df[column_name], label=f'Исходные ({column_name})')
        mean_orig = original_df[column_name].mean();
        std_orig = original_df[column_name].std()
        if std_orig > 0 and not pd.isna(std_orig):
            outliers = original_df[np.abs((original_df[column_name] - mean_orig) / std_orig) > threshold]
            if not outliers.empty: ax1.scatter(outliers.index, outliers[column_name], color='red',
                                               label=f'Выбросы (Z > {threshold})', s=50, zorder=5)

    plt.title(f'[{model_name}] Данные до удаления выбросов ({column_name})');
    plt.ylabel('Цена');
    plt.legend();
    plt.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'));
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    ax2 = plt.subplot(2, 1, 2)
    if not cleaned_df.empty and column_name in cleaned_df.columns:
        if timeframe_td is not None:
            plotted_ts_cleaned, mapped_values_cleaned = generate_plotted_times_and_values(
                cleaned_df.index, cleaned_df[column_name], timeframe_td
            )
            if len(plotted_ts_cleaned) > 0:
                ax2.plot(plotted_ts_cleaned, mapped_values_cleaned, label=f'После удаления ({column_name})',
                         color='green')
            else:
                ax2.plot(cleaned_df.index, cleaned_df[column_name],
                         label=f'После удаления ({column_name}) (fallback plot)', color='green')
        else:
            ax2.plot(cleaned_df.index, cleaned_df[column_name], label=f'После удаления ({column_name})', color='green')
    else:
        plt.text(0.5, 0.5, 'Очищенные данные отсутствуют', ha='center', va='center', transform=ax2.transAxes)

    plt.title(f'[{model_name}] Данные после удаления выбросов ({column_name})');
    plt.xlabel('Дата/время');
    plt.ylabel('Цена');
    plt.legend();
    plt.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'));
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.tight_layout();
    plt.show()


# %%
# --- БЛОК 2: Процесс предобработки данных (Pipeline) ---
def split_sequences(X_data_scaled, y_data_scaled, n_steps_in, n_steps_out, timestamps_full_data, model_name=""):
    log_prefix_seq = f"[{model_name}][SEQUENCES]"
    X, y, times_y = [], [], []
    if len(X_data_scaled) < n_steps_in + n_steps_out:
        with print_lock: print(
            f"{log_prefix_seq} Недостаточно данных для создания последовательностей: {len(X_data_scaled)} доступно, требуется {n_steps_in + n_steps_out}")
        return np.array(X), np.array(y), np.array(times_y)

    for i in range(len(X_data_scaled) - n_steps_in - n_steps_out + 1):
        X.append(X_data_scaled[i:(i + n_steps_in), :])
        y.append(y_data_scaled[(i + n_steps_in):(i + n_steps_in + n_steps_out)])
        times_y.append(timestamps_full_data[i + n_steps_in])
    return np.array(X), np.array(y), np.array(times_y)


def preprocess_data_pipeline(
        token_tinkoff, figi_instrument, days_history_candles, interval_candles,
        target_feature_col, z_threshold_outliers,
        n_steps_in_lstm, n_steps_out_lstm,
        model_name_prefix="", plot_outliers_flag=True,
        current_candle_interval_timedelta=None,
        use_news_features_flag=True
):
    log_prefix_prep = f"[{model_name_prefix}][PREPROCESS]"
    with print_lock:
        print(f"--- {log_prefix_prep} Начало процесса предобработки данных ---")

    df_raw_candles = get_tinkoff_candles(token_tinkoff, figi_instrument, days_history_candles, interval_candles,
                                         model_name=model_name_prefix)
    if df_raw_candles.empty:
        with print_lock: print(f"{log_prefix_prep} Ошибка: не удалось загрузить данные свечей. Предобработка прервана.")
        return None, None, None, None, None, None, None

    df_cleaned_candles = remove_outliers_zscore(df_raw_candles.copy(), target_feature_col,
                                                threshold=z_threshold_outliers, model_name=model_name_prefix)
    if plot_outliers_flag and current_candle_interval_timedelta:
        plot_data_with_outliers(df_raw_candles, df_cleaned_candles, target_feature_col, z_threshold_outliers,
                                model_name=model_name_prefix, timeframe_td=current_candle_interval_timedelta)

    df_features_combined = df_cleaned_candles.copy()

    if use_news_features_flag:
        with print_lock:
            print(f"{log_prefix_prep} Попытка интеграции новостных признаков...")
        if not current_candle_interval_timedelta:
            with print_lock:
                print(
                    f"{log_prefix_prep} Ошибка: current_candle_interval_timedelta не предоставлен, не могу обработать новости.")
        else:
            news_features_df = process_instrument_news(
                instrument_identifier=figi_instrument,  # Передаем FIGI, внутри будет маппинг/использование
                news_fetch_days_param=days_history_candles,
                candle_interval_timedelta=current_candle_interval_timedelta,
                p_lock=print_lock,
                model_name_prefix=model_name_prefix
            )
            if not news_features_df.empty:
                if not isinstance(df_features_combined.index,
                                  pd.DatetimeIndex): df_features_combined.index = pd.to_datetime(
                    df_features_combined.index)
                if not isinstance(news_features_df.index, pd.DatetimeIndex): news_features_df.index = pd.to_datetime(
                    news_features_df.index)

                df_features_combined = df_features_combined.merge(news_features_df, left_index=True, right_index=True,
                                                                  how='left')
                news_cols_to_fill = ['news_sentiment_mean', 'news_sentiment_sum', 'news_count', 'news_positive_count',
                                     'news_negative_count', 'news_neutral_count']
                for col in news_cols_to_fill:
                    if col in df_features_combined.columns:
                        df_features_combined[col] = df_features_combined[col].fillna(0)
                        if 'count' in col:
                            df_features_combined[col] = df_features_combined[col].astype(int)
                        else:
                            df_features_combined[col] = df_features_combined[col].astype(float)
                with print_lock:
                    print(
                        f"{log_prefix_prep} Новостные признаки успешно объединены. Размер df: {df_features_combined.shape}")
            else:
                with print_lock:
                    print(f"{log_prefix_prep} Новостные признаки не были получены или пусты. Продолжение без них.")

    df_with_ta = add_technical_indicators(df_features_combined, model_name=model_name_prefix)
    df_full_features = df_with_ta.dropna().copy()
    with print_lock:
        print(f"{log_prefix_prep} DataFrame после добавления TA и dropna(): {df_full_features.shape}")

    if df_full_features.empty or len(df_full_features) < n_steps_in_lstm + n_steps_out_lstm:
        with print_lock: print(
            f"{log_prefix_prep} Недостаточно данных после всех обработок ({len(df_full_features)} строк), требуется как минимум {n_steps_in_lstm + n_steps_out_lstm}. Предобработка прервана.")
        return None, None, None, None, None, None, df_full_features

    y_data_unscaled = df_full_features[[target_feature_col]].values
    X_feature_names = list(df_full_features.columns)
    X_data_unscaled = df_full_features[X_feature_names].values

    with print_lock:
        print(f"{log_prefix_prep} Количество признаков для X: {X_data_unscaled.shape[1]}")

    scaler_X = MinMaxScaler()
    X_data_scaled = scaler_X.fit_transform(X_data_unscaled)
    scaler_y = MinMaxScaler()
    y_data_scaled = scaler_y.fit_transform(y_data_unscaled)

    timestamps_for_sequences = df_full_features.index
    X_sequences, y_sequences, times_y_sequences = split_sequences(
        X_data_scaled, y_data_scaled.flatten(),
        n_steps_in_lstm, n_steps_out_lstm, timestamps_for_sequences, model_name=model_name_prefix)

    if X_sequences.shape[0] == 0:
        with print_lock: print(f"{log_prefix_prep} Ошибка: не удалось создать обучающие последовательности X,y.")
        return None, None, None, scaler_X, scaler_y, X_feature_names, df_full_features

    with print_lock:
        print(f"{log_prefix_prep} Предобработка завершена. Создано {X_sequences.shape[0]} последовательностей.")
        print(
            f"{log_prefix_prep} Форма X_sequences: {X_sequences.shape}, Форма y_sequences: {y_sequences.shape}, Форма times_y_sequences: {times_y_sequences.shape}")
        print(f"--- {log_prefix_prep} Конец процесса предобработки данных ---\n")

    return X_sequences, y_sequences, times_y_sequences, scaler_X, scaler_y, X_feature_names, df_full_features


# %%
# --- БЛОК 3: Построение, обучение и оценка LSTM модели ---
def build_lstm_model(n_steps_in, n_steps_out, n_features):
    model = Sequential([
        LSTM(70, return_sequences=True, input_shape=(n_steps_in, n_features)), Dropout(0.3),
        LSTM(70, return_sequences=False), Dropout(0.3),
        Dense(50, activation='relu'), Dense(n_steps_out)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def calculate_regression_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {m: np.nan for m in ['mse', 'rmse', 'mae', 'r2', 'mse_pct', 'rmse_pct', 'mae_pct', 'r2_pct']}

    mse_raw = mean_squared_error(y_true, y_pred)
    rmse_raw = np.sqrt(mse_raw)
    mae_raw = mean_absolute_error(y_true, y_pred)
    try:
        r2_raw = r2_score(y_true, y_pred)
    except ValueError:
        r2_raw = np.nan

    mean_y_true = np.mean(y_true)
    if np.isclose(mean_y_true, 0) or pd.isna(mean_y_true):
        mse_pct, rmse_pct, mae_pct = np.nan, np.nan, np.nan
    else:
        mae_pct = (mae_raw / abs(mean_y_true)) * 100
        rmse_pct = (rmse_raw / abs(mean_y_true)) * 100
        mse_pct = (mse_raw / (mean_y_true ** 2)) * 100 if not np.isclose(mean_y_true ** 2, 0) else np.nan
    r2_pct = r2_raw * 100 if not np.isnan(r2_raw) else np.nan

    return {'mse': mse_raw, 'rmse': rmse_raw, 'mae': mae_raw, 'r2': r2_raw,
            'mse_pct': mse_pct, 'rmse_pct': rmse_pct, 'mae_pct': mae_pct, 'r2_pct': r2_pct}


def train_fold(fold_num, train_idx, val_idx, X_data, y_target_scaled_data,
               scaler_y_obj, n_steps_in_lstm, n_steps_out_lstm, n_features_model,
               model_name_prefix="", current_n_cv_splits_val=N_CV_SPLITS):
    log_prefix_fold = f"[{model_name_prefix}][CV_FOLD]"
    with print_lock: print(
        f"\n--- {log_prefix_fold} Обучение fold {fold_num + 1}/{current_n_cv_splits_val} (признаков={n_features_model}) ---")

    model_fold = build_lstm_model(n_steps_in_lstm, n_steps_out_lstm, n_features_model)
    X_train_fold, y_train_target_scaled_fold = X_data[train_idx], y_target_scaled_data[train_idx]
    X_val_fold, y_val_target_scaled_fold = X_data[val_idx], y_target_scaled_data[val_idx]

    history_fold = model_fold.fit(X_train_fold, y_train_target_scaled_fold,
                                  validation_data=(X_val_fold, y_val_target_scaled_fold),
                                  epochs=50, batch_size=32, verbose=0,
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    preds_val_target_scaled_fold = model_fold.predict(X_val_fold, verbose=0)
    y_val_inv_1step_fold = scaler_y_obj.inverse_transform(y_val_target_scaled_fold[:, 0].reshape(-1, 1)).flatten()
    preds_val_inv_1step_fold = scaler_y_obj.inverse_transform(
        preds_val_target_scaled_fold[:, 0].reshape(-1, 1)).flatten()
    fold_metrics_dict = calculate_regression_metrics(y_val_inv_1step_fold, preds_val_inv_1step_fold)

    with print_lock:
        print(f"{log_prefix_fold} Результаты fold {fold_num + 1} (оценка на 1-м шаге прогноза):")
        for name, val in fold_metrics_dict.items():
            unit = " %" if name.endswith('_pct') else ""
            print(f"  {name.upper()}: {val:.4f}{unit if not pd.isna(val) else ''}")
    return {'model': model_fold, 'metrics': fold_metrics_dict, 'history': history_fold, 'fold': fold_num}


def cross_validate_lstm(X_sequences_cv, y_sequences_cv, scaler_y_obj,
                        n_splits_cv, n_steps_in_lstm, n_steps_out_lstm, n_features_model,
                        model_name_prefix="", max_workers_cv=None):
    log_prefix_cv_main = f"[{model_name_prefix}][CROSS_VALIDATION]"
    effective_max_workers_cv = 0
    if n_splits_cv >= 2:
        default_cpu_workers = (os.cpu_count() or 1) * 2
        effective_max_workers_cv = min(n_splits_cv,
                                       max_workers_cv if max_workers_cv is not None else default_cpu_workers)
        effective_max_workers_cv = max(1, effective_max_workers_cv)
    with print_lock:
        print(
            f"\n--- {log_prefix_cv_main} Начало ({n_splits_cv} фолдов, до {effective_max_workers_cv} параллельных потоков) ---")

    if n_splits_cv < 2:
        with print_lock: print(
            f"{log_prefix_cv_main} Предупреждение: n_splits_cv = {n_splits_cv} < 2. Кросс-валидация не будет выполнена.")
        return None, {}

    kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=RANDOM_STATE)
    fold_results_list = []
    futures_list = []

    with ThreadPoolExecutor(max_workers=effective_max_workers_cv,
                            thread_name_prefix=f"{model_name_prefix}_CVFold") as executor:
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_sequences_cv, y_sequences_cv)):
            futures_list.append(executor.submit(train_fold, fold_num, train_idx, val_idx,
                                                X_sequences_cv, y_sequences_cv, scaler_y_obj,
                                                n_steps_in_lstm, n_steps_out_lstm, n_features_model,
                                                model_name_prefix, current_n_cv_splits_val=n_splits_cv))
        for future_item in as_completed(futures_list):
            try:
                fold_results_list.append(future_item.result())
            except Exception as e_fold:
                with print_lock:
                    print(f"{log_prefix_cv_main} КРИТИЧЕСКАЯ ОШИБКА в одном из фолдов CV: {e_fold}")
                    import traceback;
                    print(traceback.format_exc())

    if not fold_results_list:
        with print_lock: print(f"{log_prefix_cv_main} Кросс-валидация не дала результатов.")
        return None, {}

    fold_results_list.sort(key=lambda r: r['fold'])
    best_fold_result_dict = min(fold_results_list, key=lambda r: r['metrics'].get('mse', float('inf')))

    with print_lock:
        print(f"\n--- {log_prefix_cv_main} Завершена ---")
        print(
            f"{log_prefix_cv_main} Лучшая модель выбрана из фолда {best_fold_result_dict['fold'] + 1} (на основе MSE: {best_fold_result_dict['metrics'].get('mse', float('inf')):.4f})")
        print(f"{log_prefix_cv_main} Метрики лучшей модели (на её валидационном наборе, 1-й шаг прогноза):")
        for name, val in best_fold_result_dict['metrics'].items():
            unit = " %" if name.endswith('_pct') else ""
            print(f"  {name.upper()}: {val:.4f}{unit if not pd.isna(val) else ''}")
    return best_fold_result_dict['model'], best_fold_result_dict['metrics']


def final_test_and_plot(model_final, X_test_data, y_test_target_scaled_data, scaler_y_obj,
                        times_y_test_data, n_steps_out_val, target_feature_name_val, model_name_prefix="",
                        timeframe_td_for_plot=None):
    log_prefix_final_test = f"[{model_name_prefix}][FINAL_TEST]"
    with print_lock:
        print(f"\n--- {log_prefix_final_test} Финальное тестирование модели ---")

    if X_test_data.shape[0] == 0:
        with print_lock: print(f"{log_prefix_final_test} Тестовая выборка пуста. Финальный тест не проводится.")
        return

    preds_test_target_scaled_all = model_final.predict(X_test_data, verbose=0)
    y_test_inv_1step = scaler_y_obj.inverse_transform(y_test_target_scaled_data[:, 0].reshape(-1, 1)).flatten()
    preds_test_inv_1step = scaler_y_obj.inverse_transform(preds_test_target_scaled_all[:, 0].reshape(-1, 1)).flatten()
    test_metrics_1step_dict = calculate_regression_metrics(y_test_inv_1step, preds_test_inv_1step)

    with print_lock:
        print(f"{log_prefix_final_test} Метрики на тестовой выборке (для 1-го шага прогноза):")
        for name, val in test_metrics_1step_dict.items():
            unit = " %" if name.endswith('_pct') else "";
            print(f"  {name.upper()}: {val:.4f}{unit if not pd.isna(val) else ''}")

    result_df_1step = pd.DataFrame({'datetime': times_y_test_data, 'actual_close_1step': y_test_inv_1step,
                                    'predicted_close_1step': preds_test_inv_1step}).set_index('datetime').sort_index()

    plt.figure(figsize=(15, 6));
    ax1_test = plt.gca()
    if timeframe_td_for_plot is not None:
        plotted_times_1s, _ = generate_plotted_times_and_values(result_df_1step.index,
                                                                result_df_1step['actual_close_1step'].values,
                                                                timeframe_td_for_plot)
        if len(plotted_times_1s) == len(result_df_1step):
            ax1_test.plot(plotted_times_1s, result_df_1step['actual_close_1step'].values,
                          label=f'Факт ({target_feature_name_val}, 1-й шаг)')
            ax1_test.plot(plotted_times_1s, result_df_1step['predicted_close_1step'].values,
                          label=f'Прогноз ({target_feature_name_val}, 1-й шаг)', linestyle='--')
        else:
            ax1_test.plot(result_df_1step.index, result_df_1step['actual_close_1step'],
                          label=f'Факт ({target_feature_name_val}, 1-й шаг) (fallback plot)')
            ax1_test.plot(result_df_1step.index, result_df_1step['predicted_close_1step'],
                          label=f'Прогноз ({target_feature_name_val}, 1-й шаг) (fallback plot)', linestyle='--')
    else:
        ax1_test.plot(result_df_1step.index, result_df_1step['actual_close_1step'],
                      label=f'Факт ({target_feature_name_val}, 1-й шаг)')
        ax1_test.plot(result_df_1step.index, result_df_1step['predicted_close_1step'],
                      label=f'Прогноз ({target_feature_name_val}, 1-й шаг)', linestyle='--')

    mae_pct_1step = test_metrics_1step_dict.get('mae_pct', float('nan'));
    rmse_pct_1step = test_metrics_1step_dict.get('rmse_pct', float('nan'))
    title_parts_1step = [f"[{model_name_prefix}] Факт/Прогноз ({target_feature_name_val}, 1-й шаг) на тест. выборке"]
    metrics_str_parts_1step = []
    if not pd.isna(mae_pct_1step): metrics_str_parts_1step.append(f"MAE: {mae_pct_1step:.2f}%")
    if not pd.isna(rmse_pct_1step): metrics_str_parts_1step.append(f"RMSE: {rmse_pct_1step:.2f}%")
    if metrics_str_parts_1step: title_parts_1step.append(" | ".join(metrics_str_parts_1step))
    plt.title("\n".join(title_parts_1step))
    plt.xlabel("Дата/время");
    plt.ylabel(f"Цена ({target_feature_name_val})")
    ax1_test.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'));
    ax1_test.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    plt.xticks(rotation=45);
    plt.legend();
    plt.grid(True);
    plt.tight_layout();
    plt.show()

    # Блок для оценки и графика ПОСЛЕДНЕГО шага прогноза остается закомментированным, как в оригинале
    # if n_steps_out_val > 1:
    #     ...
    pass


# %%
# --- БЛОК 4: Рекурсивное прогнозирование будущего ---
def predict_future_recursive(
        model_to_use, initial_X_sequence_scaled_val, scaler_X_obj, scaler_y_obj,
        all_X_feature_names_list, target_feature_name_val, num_total_future_steps_to_predict,
        n_steps_in_model_val, n_steps_out_model_val, last_known_full_data_df_val,
        candle_interval_timedelta_val, model_name_prefix=""):
    log_prefix_future = f"[{model_name_prefix}][FUTURE_PREDICT]"
    with print_lock:
        print(f"\n{log_prefix_future} Запуск рекурсивного прогноза на {num_total_future_steps_to_predict} шагов...")

    all_future_preds_y_inv_flat_list = []
    current_input_X_scaled_arr = initial_X_sequence_scaled_val.copy()
    history_df_unscaled = last_known_full_data_df_val.iloc[-n_steps_in_model_val:].copy()
    if not isinstance(history_df_unscaled.index, pd.DatetimeIndex): history_df_unscaled.index = pd.to_datetime(
        history_df_unscaled.index)
    last_real_timestamp_val = history_df_unscaled.index[-1]

    for _ in range(int(np.ceil(num_total_future_steps_to_predict / n_steps_out_model_val))):
        if len(all_future_preds_y_inv_flat_list) >= num_total_future_steps_to_predict: break

        predicted_y_scaled_chunk_arr = model_to_use.predict(current_input_X_scaled_arr, verbose=0)
        predicted_y_inv_chunk_arr = scaler_y_obj.inverse_transform(predicted_y_scaled_chunk_arr).flatten()

        num_to_add_this_iteration = min(len(predicted_y_inv_chunk_arr),
                                        num_total_future_steps_to_predict - len(all_future_preds_y_inv_flat_list))
        all_future_preds_y_inv_flat_list.extend(predicted_y_inv_chunk_arr[:num_to_add_this_iteration])
        if len(all_future_preds_y_inv_flat_list) >= num_total_future_steps_to_predict: break

        new_rows_for_history_list = []
        current_synthetic_timestamp = history_df_unscaled.index[-1]
        for i in range(len(predicted_y_inv_chunk_arr)):
            current_synthetic_timestamp += candle_interval_timedelta_val
            predicted_close_for_new_row = predicted_y_inv_chunk_arr[i]
            new_row_dict = {'open': predicted_close_for_new_row, 'high': predicted_close_for_new_row,
                            'low': predicted_close_for_new_row, 'close': predicted_close_for_new_row,
                            'volume': history_df_unscaled['volume'].iloc[
                                -1] if 'volume' in history_df_unscaled.columns else 0}
            for col_name_in_X in all_X_feature_names_list:
                if col_name_in_X not in new_row_dict: new_row_dict[col_name_in_X] = np.nan
            new_rows_for_history_list.append(pd.DataFrame(new_row_dict, index=[current_synthetic_timestamp]))
        if not new_rows_for_history_list: continue

        history_df_unscaled_extended_ohlcv = pd.concat([history_df_unscaled] + new_rows_for_history_list)
        lookback_for_ta_recalc = max(
            [p for p in SMA_PERIODS if isinstance(p, int)] + [MACD_SLOW + MACD_SIGNAL, ATR_PERIOD, BBANDS_PERIOD,
                                                              RSI_PERIOD, STOCH_K + STOCH_SMOOTH_K, 50]) + 20
        start_idx_for_ta_recalc = max(0, len(history_df_unscaled_extended_ohlcv) - (
                    lookback_for_ta_recalc + n_steps_out_model_val))
        df_for_ta_recalculation = history_df_unscaled_extended_ohlcv.iloc[start_idx_for_ta_recalc:].copy()
        news_cols_in_history = [col for col in all_X_feature_names_list if 'news_' in col]
        for ncol in news_cols_in_history:
            if ncol in df_for_ta_recalculation.columns: df_for_ta_recalculation[ncol] = df_for_ta_recalculation[
                ncol].fillna(0)
        df_with_new_ta_recalculated = add_technical_indicators(df_for_ta_recalculation,
                                                               model_name=f"{model_name_prefix}-future_TA")

        if df_with_new_ta_recalculated.empty or len(df_with_new_ta_recalculated) < n_steps_in_model_val:
            with print_lock: print(
                f"{log_prefix_future} Ошибка: недостаточно данных ({len(df_with_new_ta_recalculated)}) после пересчета TA для будущего прогноза. Прогноз остановлен.")
            break
        history_df_unscaled = df_with_new_ta_recalculated.copy()

        new_X_unscaled_for_input_df = history_df_unscaled[all_X_feature_names_list].iloc[-n_steps_in_model_val:]
        if len(new_X_unscaled_for_input_df) < n_steps_in_model_val:
            with print_lock: print(
                f"{log_prefix_future} Ошибка: недостаточно данных ({len(new_X_unscaled_for_input_df)}) для формирования X_input после пересчета TA. Прогноз остановлен.")
            break
        new_X_unscaled_values_arr = new_X_unscaled_for_input_df.values
        if np.isnan(new_X_unscaled_values_arr).any():
            temp_df_fill_nan = pd.DataFrame(new_X_unscaled_values_arr, columns=all_X_feature_names_list)
            temp_df_fill_nan = temp_df_fill_nan.fillna(method='ffill').fillna(method='bfill').fillna(0)
            new_X_unscaled_values_arr = temp_df_fill_nan.values
            if np.isnan(new_X_unscaled_values_arr).any() and (_ % 5 == 0):
                with print_lock: print(
                    f"{log_prefix_future} Предупреждение: NaN остались в X_input даже после ffill/bfill/0 на шаге {_}.")
        new_X_scaled_values_arr = scaler_X_obj.transform(new_X_unscaled_values_arr)
        current_input_X_scaled_arr = new_X_scaled_values_arr.reshape(1, n_steps_in_model_val, -1)

    future_predictions_array = np.array(all_future_preds_y_inv_flat_list)
    future_timestamps_array = pd.to_datetime([last_real_timestamp_val + candle_interval_timedelta_val * (i + 1) for i in
                                              range(len(future_predictions_array))])
    with print_lock:
        print(
            f"{log_prefix_future} Рекурсивный прогноз завершен. Предсказано {len(future_predictions_array)} шагов из {num_total_future_steps_to_predict} запрошенных.")
    return future_predictions_array, future_timestamps_array


# %%
# --- БЛОК 5: Основной поток выполнения для одной модели (конфигурации) ---
def run_model_pipeline(model_config_dict):
    model_name = model_config_dict["name"]
    print_lock.acquire();
    print(f"\n\n{'=' * 30} НАЧАЛО ОБРАБОТКИ МОДЕЛИ: {model_name} {'=' * 30}");
    print_lock.release()

    days_history_for_data = model_config_dict["days_history"]
    candle_timeframe_enum = model_config_dict["candle_timeframe"]
    n_steps_in_for_lstm = model_config_dict["n_steps_in"]
    n_steps_out_for_lstm = model_config_dict["n_steps_out"]
    future_predictions_num_steps = model_config_dict["future_predictions"]
    use_news_features_for_model = model_config_dict.get("use_news_features", True)
    current_tf_timedelta = get_timedelta_from_candle_interval(candle_timeframe_enum)

    X_sequences_all, y_sequences_all, times_y_all, scaler_X_obj, scaler_y_obj, x_all_feature_names, df_full_features_processed = preprocess_data_pipeline(
        token_tinkoff=TINKOFF_API_TOKEN, figi_instrument=INSTRUMENT_FIGI, days_history_candles=days_history_for_data,
        interval_candles=candle_timeframe_enum, target_feature_col=FEATURE_COLUMN,
        z_threshold_outliers=Z_SCORE_THRESHOLD,
        n_steps_in_lstm=n_steps_in_for_lstm, n_steps_out_lstm=n_steps_out_for_lstm, model_name_prefix=model_name,
        plot_outliers_flag=False, current_candle_interval_timedelta=current_tf_timedelta,
        use_news_features_flag=use_news_features_for_model
    )

    if X_sequences_all is None or X_sequences_all.shape[0] == 0:
        with print_lock: print(
            f"[{model_name}][PIPELINE_ERROR] Не удалось подготовить данные. Пропуск модели {model_name}.")
        print_lock.acquire();
        print(f"{'=' * 30} ЗАВЕРШЕНИЕ ОБРАБОТКИ МОДЕЛИ: {model_name} (с ошибкой данных) {'=' * 30}");
        print_lock.release()
        return

    n_total_features_for_X = X_sequences_all.shape[2]
    min_samples_for_split = int(1 / TEST_SPLIT_SIZE) if TEST_SPLIT_SIZE > 0 else float('inf')
    if X_sequences_all.shape[0] < min_samples_for_split and X_sequences_all.shape[0] > N_CV_SPLITS * 2:
        with print_lock:
            print(
                f"[{model_name}][DATA_SPLIT] Недостаточно данных ({X_sequences_all.shape[0]}) для test_split. Используем все для train/CV.")
        X_trainval, X_test = X_sequences_all, np.array([])
        y_trainval_target_scaled, y_test_target_scaled = y_sequences_all, np.array([])
        times_y_trainval, times_y_test = times_y_all, np.array([])
    elif X_sequences_all.shape[0] <= N_CV_SPLITS * 2:
        with print_lock:
            print(
                f"[{model_name}][DATA_SPLIT] СЛИШКОМ МАЛО ДАННЫХ ({X_sequences_all.shape[0]}). Пропуск модели {model_name}.")
        print_lock.acquire();
        print(f"{'=' * 30} ЗАВЕРШЕНИЕ ОБРАБОТКИ МОДЕЛИ: {model_name} (слишком мало данных) {'=' * 30}");
        print_lock.release()
        return
    else:
        X_trainval, X_test, y_trainval_target_scaled, y_test_target_scaled, times_y_trainval, times_y_test = train_test_split(
            X_sequences_all, y_sequences_all, times_y_all, test_size=TEST_SPLIT_SIZE, shuffle=False,
            random_state=RANDOM_STATE
        )
    with print_lock:
        print(f"\n[{model_name}][DATA_SPLIT] Размеры выборок: Обуч/валид: {X_trainval.shape}, Тест: {X_test.shape}")

    final_trained_model = None
    current_cv_splits_to_use = N_CV_SPLITS
    if X_trainval.shape[0] == 0:
        with print_lock:
            print(f"[{model_name}][TRAINING] Выборка X_trainval пуста. Пропуск обучения.")
    elif X_trainval.shape[0] < current_cv_splits_to_use:
        if X_trainval.shape[0] >= 2:
            actual_cv_splits = max(2, X_trainval.shape[0] // 2 if X_trainval.shape[0] >= 4 else X_trainval.shape[0])
            if actual_cv_splits < 2: actual_cv_splits = 0
            with print_lock:
                print(
                    f"[{model_name}][TRAINING] Мало данных для {current_cv_splits_to_use} CV-фолдов. Попытка {actual_cv_splits} фолдов.")
            current_cv_splits_to_use = actual_cv_splits
        else:
            current_cv_splits_to_use = 0
            with print_lock:
                print(f"[{model_name}][TRAINING] Слишком мало данных ({X_trainval.shape[0]}) для CV/обучения. Пропуск.")

    if current_cv_splits_to_use >= 2:
        best_model_from_cv, _ = cross_validate_lstm(X_trainval, y_trainval_target_scaled, scaler_y_obj,
                                                    current_cv_splits_to_use, n_steps_in_for_lstm, n_steps_out_for_lstm,
                                                    n_total_features_for_X, model_name_prefix=model_name,
                                                    max_workers_cv=MAX_CV_WORKERS)
        final_trained_model = best_model_from_cv
    elif X_trainval.shape[0] > 0:
        with print_lock:
            print(f"[{model_name}][TRAINING] Обучение одной модели без CV ({X_trainval.shape[0]} семплов)...")
        final_trained_model = build_lstm_model(n_steps_in_for_lstm, n_steps_out_for_lstm, n_total_features_for_X)
        temp_X_train_single, temp_y_train_single = X_trainval, y_trainval_target_scaled
        val_data_for_single_fit = None
        if len(X_trainval) > max(10, (n_steps_in_for_lstm + n_steps_out_for_lstm)) * 1.2:
            temp_X_train_single, temp_X_val_single, temp_y_train_single, temp_y_val_single = train_test_split(
                X_trainval, y_trainval_target_scaled, test_size=0.1, shuffle=False, random_state=RANDOM_STATE)
            if len(temp_X_val_single) > 0: val_data_for_single_fit = (temp_X_val_single, temp_y_val_single)
        final_trained_model.fit(temp_X_train_single, temp_y_train_single, epochs=50, batch_size=32, verbose=0,
                                validation_data=val_data_for_single_fit, callbacks=[
                EarlyStopping(monitor='val_loss' if val_data_for_single_fit else 'loss', patience=10,
                              restore_best_weights=True)])
    else:
        with print_lock:
            print(f"[{model_name}][TRAINING] Нет данных для обучения. Модель не будет обучена.")

    if final_trained_model:
        if X_test.shape[0] > 0:
            final_test_and_plot(final_trained_model, X_test, y_test_target_scaled, scaler_y_obj, times_y_test,
                                n_steps_out_for_lstm, FEATURE_COLUMN, model_name_prefix=model_name,
                                timeframe_td_for_plot=current_tf_timedelta)
        else:
            with print_lock:
                print(f"[{model_name}][FINAL_TEST] Тестовая выборка пуста, тест не проводился.")

        if df_full_features_processed is not None and not df_full_features_processed.empty and len(
                df_full_features_processed) >= n_steps_in_for_lstm and scaler_X_obj and scaler_y_obj and x_all_feature_names:
            last_N_real_data_unscaled_X_df = df_full_features_processed[x_all_feature_names].iloc[-n_steps_in_for_lstm:]
            if last_N_real_data_unscaled_X_df.isnull().values.any():
                with print_lock:
                    print(
                        f"[{model_name}][FUTURE_PREDICT_INIT] NaN в данных для рекурсивного прогноза. Заполнение ffill/bfill/0...")
                last_N_real_data_unscaled_X_df = last_N_real_data_unscaled_X_df.fillna(method='ffill').fillna(
                    method='bfill').fillna(0)
                if last_N_real_data_unscaled_X_df.isnull().values.any():
                    with print_lock: print(
                        f"[{model_name}][FUTURE_PREDICT_INIT] ПРЕДУПРЕЖДЕНИЕ: Не удалось заполнить NaN. Прогноз может быть неточным.")
            initial_sequence_X_for_future_scaled = scaler_X_obj.transform(last_N_real_data_unscaled_X_df.values)
            initial_sequence_X_for_future_reshaped = initial_sequence_X_for_future_scaled.reshape(1,
                                                                                                  n_steps_in_for_lstm,
                                                                                                  n_total_features_for_X)

            future_preds_y_inv_arr, future_times_arr = predict_future_recursive(
                model_to_use=final_trained_model, initial_X_sequence_scaled_val=initial_sequence_X_for_future_reshaped,
                scaler_X_obj=scaler_X_obj, scaler_y_obj=scaler_y_obj, all_X_feature_names_list=x_all_feature_names,
                target_feature_name_val=FEATURE_COLUMN, num_total_future_steps_to_predict=future_predictions_num_steps,
                n_steps_in_model_val=n_steps_in_for_lstm, n_steps_out_model_val=n_steps_out_for_lstm,
                last_known_full_data_df_val=df_full_features_processed.copy(),
                candle_interval_timedelta_val=current_tf_timedelta, model_name_prefix=model_name
            )

            if future_preds_y_inv_arr is not None and len(future_preds_y_inv_arr) > 0:
                future_predictions_df = pd.DataFrame(
                    {'datetime': future_times_arr, 'predicted_value': future_preds_y_inv_arr.flatten()}).set_index(
                    'datetime')
                with print_lock:
                    print(
                        f"\n[{model_name}][FUTURE_PREDICT_RESULT] Будущие прогнозы ({len(future_preds_y_inv_arr)}/{future_predictions_num_steps} шагов):")
                    print(future_predictions_df.head())
                save_predictions_to_csv(future_predictions_df=future_predictions_df, model_name=model_name,
                                        instrument_figi=INSTRUMENT_FIGI, target_feature_name=FEATURE_COLUMN,
                                        candle_interval_name=candle_timeframe_enum.name, p_lock=print_lock)

                plt.figure(figsize=(15, 7));
                ax_fut1 = plt.gca()
                context_len = min(n_steps_in_for_lstm * 2, len(df_full_features_processed[FEATURE_COLUMN]))
                real_data_context_vals = df_full_features_processed[FEATURE_COLUMN].values[-context_len:]
                real_data_context_times = df_full_features_processed.index[-context_len:]
                plotted_ctx_times, plotted_ctx_vals = generate_plotted_times_and_values(real_data_context_times,
                                                                                        real_data_context_vals,
                                                                                        current_tf_timedelta)
                if len(plotted_ctx_times) > 0:
                    ax_fut1.plot(plotted_ctx_times, plotted_ctx_vals, label=f'Реальные ({FEATURE_COLUMN})', marker='.')
                else:
                    ax_fut1.plot(real_data_context_times, real_data_context_vals,
                                 label=f'Реальные ({FEATURE_COLUMN}) (fallback plot)', marker='.')

                if len(future_times_arr) > 0 and len(real_data_context_times) > 0:
                    combined_times_for_plot = pd.DatetimeIndex(
                        real_data_context_times.tolist() + future_times_arr.tolist())
                    combined_values_for_plot = np.concatenate(
                        (real_data_context_vals, future_preds_y_inv_arr.flatten()))
                    plotted_combined_times, _ = generate_plotted_times_and_values(combined_times_for_plot,
                                                                                  combined_values_for_plot,
                                                                                  current_tf_timedelta)
                    if len(plotted_combined_times) == len(combined_times_for_plot):
                        plotted_future_segment_times = plotted_combined_times[len(real_data_context_times):]
                        ax_fut1.plot(plotted_future_segment_times, future_preds_y_inv_arr.flatten(),
                                     label=f'Прогноз на {len(future_preds_y_inv_arr)} шагов', marker='x',
                                     linestyle='--')
                    else:
                        ax_fut1.plot(future_times_arr, future_preds_y_inv_arr.flatten(),
                                     label=f'Прогноз на {len(future_preds_y_inv_arr)} шагов (fallback plot)',
                                     marker='x', linestyle='--')
                elif len(future_times_arr) > 0:
                    ax_fut1.plot(future_times_arr, future_preds_y_inv_arr.flatten(),
                                 label=f'Прогноз ({FEATURE_COLUMN}) (no context)', marker='x', linestyle='--')
                plt.title(f'[{model_name}] Контекст и Прогноз {FEATURE_COLUMN} ({candle_timeframe_enum.name})');
                plt.xlabel('Дата/время');
                plt.ylabel(f"Цена ({FEATURE_COLUMN})")
                ax_fut1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'));
                ax_fut1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
                plt.xticks(rotation=45);
                plt.legend();
                plt.grid(True);
                plt.tight_layout();
                plt.show()
            else:
                with print_lock:
                    print(f"[{model_name}][FUTURE_PREDICT_RESULT] Рекурсивный прогноз не вернул результатов.")
        else:
            with print_lock:
                print(
                    f"[{model_name}][FUTURE_PREDICT_SKIP] Пропуск рекурсивного прогноза: недостаточно данных / нет скейлеров / нет df_full_features.")
    else:
        with print_lock:
            print(f"[{model_name}][PIPELINE_END] Модель не была обучена. Тест и прогноз пропущены.")

    print_lock.acquire();
    print(f"\n{'=' * 30} ЗАВЕРШЕНИЕ ОБРАБОТКИ МОДЕЛИ: {model_name} {'=' * 30}");
    print_lock.release()


# %%
# --- ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ ---
if __name__ == "__main__":
    with print_lock:
        print("--- Начало работы основной программы прогнозирования ---")
        if TINKOFF_API_TOKEN == "YOUR_TINKOFF_API_KEY":  # Плейсхолдер для Tinkoff токена, если он есть в config
            print(
                "ПРЕДУПРЕЖДЕНИЕ: TINKOFF_API_TOKEN не установлен (используется значение по умолчанию)! Загрузка данных Tinkoff не удастся.")

        # Проверка ключа NewsAPI
        if NEWSAPI_KEY == NEWSAPI_KEY_DEFAULT_PLACEHOLDER:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: NEWSAPI_KEY не установлен (равен плейсхолдеру '{NEWSAPI_KEY_DEFAULT_PLACEHOLDER}')! Будут использованы мок-данные для новостей.")
        elif not NEWSAPI_KEY:  # Если ключ пустая строка
            print("ПРЕДУПРЕЖДЕНИЕ: NEWSAPI_KEY пуст! Будут использованы мок-данные для новостей.")

    load_sentiment_model_if_needed(print_lock)

    model_configs_list = [
        {
            "name": "Model_1min_TF_30d_hist_5h_pred_WITH_NEWS",
            "days_history": 30,
            "candle_timeframe": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "n_steps_in": 150,
            "n_steps_out": 15,
            "future_predictions": 300,
            "use_news_features": True
        },
        {
            "name": "Model_5min_TF_60d_hist_1day_pred_WITH_NEWS",
            "days_history": 60,  # NewsAPI free tier limited to ~30 days history
            "candle_timeframe": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "n_steps_in": int(117 * 2.0),
            "n_steps_out": 117,
            "future_predictions": 117 * 2,
            "use_news_features": True
        },
        {
            "name": "Model_10min_TF_90d_hist_1day_pred_WITH_NEWS",  # NewsAPI free tier limited to ~30 days history
            "days_history": 90,
            "candle_timeframe": CandleInterval.CANDLE_INTERVAL_10_MIN,
            "n_steps_in": int(59 * 2.5),
            "n_steps_out": 59,
            "future_predictions": 59 * 2,
            "use_news_features": True
        },
    ]

    MIN_DESIRED_MODEL_WORKERS = 3
    cpu_cores_count = os.cpu_count()
    if cpu_cores_count is None:
        with print_lock:
            print(
                f"Предупреждение: Не удалось определить CPU ядер. Будет использовано до {MIN_DESIRED_MODEL_WORKERS} потоков.")
        max_cpu_based_model_workers = MIN_DESIRED_MODEL_WORKERS
    else:
        max_cpu_based_model_workers = cpu_cores_count
        with print_lock:
            print(f"Обнаружено {cpu_cores_count} CPU ядер.")

    effective_model_workers = min(len(model_configs_list), max_cpu_based_model_workers)
    MAX_MODEL_WORKERS_TO_RUN = min(effective_model_workers,
                                   MIN_DESIRED_MODEL_WORKERS) if MIN_DESIRED_MODEL_WORKERS > 0 else effective_model_workers
    MAX_MODEL_WORKERS_TO_RUN = max(1, MAX_MODEL_WORKERS_TO_RUN) if len(model_configs_list) > 0 else 0

    with print_lock:
        print(f"INFO: Запуск обучения для {len(model_configs_list)} моделей.")
        if MIN_DESIRED_MODEL_WORKERS > 0: print(f"INFO: Запрошено до {MIN_DESIRED_MODEL_WORKERS} параллельных потоков.")
        print(f"INFO: Максимальное количество одновременно обучаемых моделей: {MAX_MODEL_WORKERS_TO_RUN}")
        if MAX_MODEL_WORKERS_TO_RUN > 1 and gpus:
            print("INFO: ВНИМАНИЕ! Если используется GPU, убедитесь, что TensorFlow настроен для роста памяти GPU.")

    if len(model_configs_list) > 0 and MAX_MODEL_WORKERS_TO_RUN > 0:
        # Создаем карту для отслеживания конфигурации каждого future, чтобы логировать ошибки правильно
        future_to_config_map = {}
        with ThreadPoolExecutor(max_workers=MAX_MODEL_WORKERS_TO_RUN, thread_name_prefix="ModelPipeline") as executor:
            for config in model_configs_list:
                future = executor.submit(run_model_pipeline, config)
                future_to_config_map[future] = config

            for future_model_run in as_completed(future_to_config_map):
                completed_config = future_to_config_map[future_model_run]
                model_name_for_error_log = completed_config.get("name", "unknown_model")
                try:
                    future_model_run.result()
                except Exception as e_model_pipeline:
                    with print_lock:
                        print(
                            f"КРИТИЧЕСКАЯ ОШИБКА (модель: {model_name_for_error_log}): Обработка модели завершилась с исключением: {e_model_pipeline}")
                        import traceback

                        print(traceback.format_exc())
    elif len(model_configs_list) == 0:
        with print_lock:
            print("INFO: Нет конфигураций моделей для обучения. Завершение.")
    else:
        with print_lock:
            print(
                f"INFO: MAX_MODEL_WORKERS_TO_RUN ({MAX_MODEL_WORKERS_TO_RUN}) или количество моделей ({len(model_configs_list)}) равно 0. Обучение не будет запущено.")

    with print_lock:
        print("\n--- Работа всей программы прогнозирования завершена ---")