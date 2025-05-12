from fastapi import FastAPI, HTTPException, Query  # Query можно не импортировать явно, если просто тип указан
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tinkoff.invest import Client, CandleInterval, Quotation, MoneyValue
from tinkoff.invest.caching.market_data_cache.cache import MarketDataCache
from tinkoff.invest.caching.market_data_cache.cache_settings import MarketDataCacheSettings
from tinkoff.invest.utils import now

import subprocess
import pandas as pd
# Убрали дубликат from fastapi.responses import JSONResponse

from datetime import timedelta
from pathlib import Path
import json
import traceback  # Для детального логирования ошибок
import os  # Для проверки существования файла

from config import TOKEN

import sys
sys.stdout.reconfigure(line_buffering=True)


API_TOKEN = TOKEN
ACCOUNT_ID = "2094055480"

app = FastAPI(title="Tinkoff Portfolio Stream API")
app.mount("/static", StaticFiles(directory="static"), name="static")


def quotation_to_float(q: Quotation) -> float:
    if q is None:
        return 0.0
    return q.units + q.nano / 1_000_000_000.0


def money_value_to_float(mv: MoneyValue) -> float:
    if mv is None:
        return 0.0
    return mv.units + mv.nano / 1_000_000_000.0


@app.get("/", response_class=FileResponse)
def index():
    return "static/portfolio.html"


def portfolio_generator(token: str, account_id: str):
    with Client(token) as client:
        try:
            port = client.operations.get_portfolio(account_id=account_id)
            avg_price_map = {}
            for p_pos in port.positions:
                avg_price_map[p_pos.figi] = money_value_to_float(p_pos.average_position_price)

            resp = client.users.get_accounts()
            valid_accounts = [a.id for a in resp.accounts]
            if account_id not in valid_accounts:
                # Если ID не валиден, можно отправить ошибку или пустой стрим.
                # Для примера, отправим событие ошибки и завершим.
                error_payload = {"type": "error", "message": f"Account ID {account_id} not found or invalid."}
                yield f"data: {json.dumps(error_payload)}\n\n"
                print(f"Error: Account ID {account_id} not found. Valid accounts: {valid_accounts}")
                return

            stream_accounts = [account_id]
            stream = client.operations_stream.positions_stream(
                accounts=stream_accounts, with_initial_positions=True  # Запрашиваем начальные позиции
            )

            initial_data_sent = False
            for update in stream:
                if hasattr(update, "initial_positions") and update.initial_positions and not initial_data_sent:

                    init = update.initial_positions
                    items = []
                    total_value_portfolio = 0.0

                    for s in init.securities:
                        figi = s.figi
                        balance = s.balance
                        avg = avg_price_map.get(figi, 0.0)

                        # Получаем текущую цену
                        # Для стрима лучше избегать множественных синхронных запросов внутри цикла
                        # Но для начальной загрузки это приемлемо, если инструментов немного
                        # В идеале, цены тоже должны стримиться или запрашиваться батчем
                        curr_price = 0.0
                        try:
                            last_prices_response = client.market_data.get_last_prices(figi=[figi])
                            if last_prices_response.last_prices:
                                price_quotation = last_prices_response.last_prices[0].price
                                curr_price = quotation_to_float(price_quotation)
                        except Exception as e_price:
                            print(f"Could not get price for {figi}: {e_price}")
                            # Можно установить цену в 0 или пропустить инструмент

                        # Получаем информацию об инструменте
                        name = "Unknown"
                        ticker = "N/A"
                        try:
                            instr_info = client.instruments.get_instrument_by(id_type=1, class_code="",
                                                                              id=figi)  # id_type=1 для FIGI
                            if instr_info and instr_info.instrument:
                                instr = instr_info.instrument
                                name = instr.name
                                ticker = instr.ticker
                        except Exception as e_instr:
                            print(f"Could not get instrument info for {figi}: {e_instr}")

                        value = balance * curr_price
                        total_value_portfolio += value
                        pl = value - (balance * avg) if avg > 0 else value

                        items.append({
                            "figi": figi,
                            "ticker": ticker,
                            "name": name,
                            "quantity": balance,
                            "avg_price": avg,
                            "curr_price": curr_price,
                            "value": value,
                            "pl": pl
                        })

                    # Общая стоимость портфеля из get_portfolio (если доступна и актуальна)
                    # или рассчитанная сумма. Рассчитанная сумма по текущим ценам может быть точнее.
                    # total_portfolio_value_from_api = money_value_to_float(port.total_amount_portfolio)

                    payload = {"type": "initial", "data": {"positions": items, "total": total_value_portfolio}}
                    yield f"data: {json.dumps(payload)}\n\n"
                    initial_data_sent = True
                    # Если нужен только initial snapshot, можно здесь сделать break
                    # break


                elif hasattr(update, "position_update") and update.position_update:
                    # Это более сложная часть, требующая обновления данных на клиенте
                    # Пока оставим только initial
                    pass

                # Можно добавить обработку других типов сообщений из PositionsStream, если нужно

            # Если стрим завершился, а начальные данные не были отправлены (маловероятно с with_initial_positions=True)
            if not initial_data_sent:
                print("Warning: Stream ended before initial positions could be sent.")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Could not retrieve initial portfolio data.'})}\n\n"


        except Exception as e:
            print(f"Error in portfolio_generator: {e}")
            traceback.print_exc()
            error_payload = {"type": "error", "message": f"Server error in portfolio stream: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/portfolio/stream")
def stream_portfolio():
    # Токен и ID счета можно получать из запроса, кук, заголовков и т.д. для реального приложения
    # Здесь используем глобальные для простоты
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="API_TOKEN not configured on server.")
    if not ACCOUNT_ID:
        raise HTTPException(status_code=500, detail="ACCOUNT_ID not configured on server.")

    return StreamingResponse(
        portfolio_generator(API_TOKEN, ACCOUNT_ID),
        media_type="text/event-stream"
    )
    # Обертка try-except здесь может быть излишней, если ошибки обрабатываются в генераторе


@app.get("/candles")
def get_candles(figi: str, days: int = 30):
    try:
        with Client(API_TOKEN) as client:
            # Использование кэша опционально, но хорошо для производительности
            # cache = MarketDataCache(
            #     settings=MarketDataCacheSettings(base_cache_dir=Path("cache")),
            #     services=client
            # )
            start_date = now() - timedelta(days=days)

            # candles_data = list(cache.get_all_candles(
            candles_data = list(client.get_all_candles(  # Используем напрямую без кеша для простоты
                figi=figi,
                from_=start_date,
                to=now(),  # Явно указываем to
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))

            if not candles_data:
                return JSONResponse(content=[])  # Возвращаем пустой массив, а не JSONResponse([])

            response_candles = []
            for c in candles_data:
                if c.time and c.close is not None:  # Проверяем, что close не None
                    response_candles.append({
                        "time": c.time.isoformat(),
                        "open": quotation_to_float(c.open),
                        "close": quotation_to_float(c.close),
                        "high": quotation_to_float(c.high),  # Добавим high/low если нужны
                        "low": quotation_to_float(c.low),
                        "volume": c.volume
                    })
            return JSONResponse(content=response_candles)
    except Exception as e:
        print(f"Error in get_candles for FIGI {figi}:")
        traceback.print_exc()
        # В detail передаем строку, а не сам объект ошибки
        raise HTTPException(status_code=500, detail=f"Internal server error processing candles for {figi}: {str(e)}")


@app.get("/forecast")
def forecast(figi: str = Query(..., description="FIGI инструмента для прогнозирования")):
    print(f"Received forecast request for FIGI: {figi}")
    if not figi:
        raise HTTPException(status_code=400, detail="FIGI parameter is required.")

    forecast_file_path = "forecast_result.csv" # Relative to where main.py runs

    try:
        if os.path.exists(forecast_file_path):
            os.remove(forecast_file_path)

        # Ensure 'python3' is correct. Consider using sys.executable for robustness
        # if test_validation.py is meant to run with the same Python interpreter.
        # For simplicity, 'python3' is often fine if it's in the system PATH.
        script_path = "test_validation.py" # Assumes it's in the same directory or accessible via PATH

        # Check if script exists to give a clearer error
        if not os.path.exists(script_path):
            print(f"Error: Training script '{script_path}' not found.")
            raise HTTPException(status_code=500, detail=f"Model training script '{script_path}' not found.")

        print(f"Running command: ['python3', '{script_path}', '{figi}']")
        process = subprocess.run(
            ["python3", script_path, figi],
            capture_output=True, text=True, check=False, timeout=600  # 10 min timeout
        )

        print(f"--- Subprocess STDOUT for FIGI {figi} ---")
        print(process.stdout if process.stdout.strip() else "<empty stdout>")
        print(f"--- Subprocess STDERR for FIGI {figi} ---")
        print(process.stderr if process.stderr.strip() else "<empty stderr>")
        print(f"--- Subprocess Return Code for FIGI {figi}: {process.returncode} ---")


        if process.returncode != 0:
            error_detail = f"Model training script failed for {figi} with exit code {process.returncode}."
            if process.stderr:
                error_detail += f" Stderr: {process.stderr[:500]}" # Limit stderr length
            print(f"Error during model training script for FIGI {figi}: Return code {process.returncode}")
            # No need to print stdout/stderr here again, already done above
            raise HTTPException(status_code=500, detail=error_detail)

        if not os.path.exists(forecast_file_path):
            print(f"Forecast file '{forecast_file_path}' not found after script execution for FIGI {figi}.")
            # Output from script was already printed above
            raise HTTPException(status_code=500,
                                detail=f"Forecast result file not found for {figi} after script execution. Check server logs for script output.")

        df = pd.read_csv(forecast_file_path, index_col=0, parse_dates=True)

        if "predicted_close" not in df.columns:
            print(f"'predicted_close' column not found in {forecast_file_path} for FIGI {figi}.")
            raise HTTPException(status_code=500,
                                detail=f"Forecast data is malformed for {figi}: 'predicted_close' column missing.")

        # Allow empty forecast if the script decided no prediction could be made
        # if df.empty:
        #     print(f"Forecast data is empty for FIGI {figi}.")
        #     # Return an empty forecast, not necessarily an error
        #     return JSONResponse(content={"times": [], "prices": []})


        data = {
            "times": [ts.isoformat() for ts in df.index],
            "prices": df["predicted_close"].tolist() if not df.empty else []
        }
        return JSONResponse(content=data)

    except subprocess.TimeoutExpired:
        print(f"Timeout running model training script for FIGI {figi}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model training script timed out for {figi}.")
    except FileNotFoundError as e: # If 'python3' or script itself is not found by subprocess
        print(f"Script execution error (FileNotFoundError): {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model training script execution error: {e.strerror}")
    except Exception as e:
        print(f"Unhandled error in forecast endpoint for FIGI {figi}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error processing forecast for {figi}: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)