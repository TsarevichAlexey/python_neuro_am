from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tinkoff.invest import Client, CandleInterval, Quotation, MoneyValue  # Добавлены Quotation, MoneyValue
from tinkoff.invest.caching.market_data_cache.cache import MarketDataCache
from tinkoff.invest.caching.market_data_cache.cache_settings import MarketDataCacheSettings
from tinkoff.invest.utils import now
# from pydantic import BaseModel # BaseModel не используется, можно убрать
from datetime import timedelta
from pathlib import Path
import os, json
import traceback  # Для детального логирования ошибок

from config import TOKEN

API_TOKEN = TOKEN
ACCOUNT_ID = "2073035222"  # Замените на ваш реальный ID или уберите, если не нужен жестко

app = FastAPI(title="Tinkoff Portfolio Stream API")
app.mount("/static", StaticFiles(directory="static"), name="static")


def quotation_to_float(q: Quotation) -> float:
    """Конвертирует объект Quotation в float."""
    if q is None:
        return 0.0
    return q.units + q.nano / 1_000_000_000.0


def money_value_to_float(mv: MoneyValue) -> float:
    """Конвертирует объект MoneyValue в float."""
    if mv is None:
        return 0.0
    return mv.units + mv.nano / 1_000_000_000.0


@app.get("/", response_class=FileResponse)
def index():
    return "static/portfolio.html"


def portfolio_generator(token: str, account_id: str):
    """SSE: отдаём сразу всю корзину со всеми нужными полями."""
    with Client(token) as client:
        port = client.operations.get_portfolio(account_id=account_id)
        avg_price_map = {}
        for p_pos in port.positions:
            avg_price_map[p_pos.figi] = money_value_to_float(p_pos.average_position_price)

        resp = client.users.get_accounts()
        # Если account_id не указан или не найден, можно взять первый счет или обработать ошибку
        # Здесь предполагается, что account_id всегда валиден и есть в списке
        # Для большей надежности можно добавить проверку
        valid_accounts = [a.id for a in resp.accounts]
        if account_id not in valid_accounts and valid_accounts:
            print(f"Warning: Account ID {account_id} not found. Using first available: {valid_accounts[0]}")
            # account_id = valid_accounts[0] # или выбросить ошибку / вернуть пустой стрим

        # Если ACCOUNT_ID жестко задан и может быть не в списке, то для stream лучше передавать [account_id]
        # Однако, API может требовать, чтобы ID был из get_accounts.
        # Для positions_stream обычно передают список ID, которые пользователь хочет слушать.
        # Изначально было accounts=accounts, но если мы хотим только один, то [account_id]
        stream_accounts = [account_id]  # Фокусируемся на указанном счете

        stream = client.operations_stream.positions_stream(
            accounts=stream_accounts, with_initial_positions=True
        )
        for update in stream:
            if not update.initial_positions:  # Обрабатываем только первое сообщение с полным портфелем
                continue

            init = update.initial_positions
            items = []
            total_value_portfolio = 0.0  # Используем другое имя, чтобы не конфликтовать с total_value от get_portfolio

            # Получаем общую стоимость из get_portfolio для согласованности, если нужно
            # total_value_from_get_portfolio = money_value_to_float(port.total_amount_portfolio)

            for s in init.securities:  # s это PositionsSecurities
                figi = s.figi
                balance = s.balance  # Это int64, количество штук
                avg = avg_price_map.get(figi, 0.0)

                last_prices_response = client.market_data.get_last_prices(figi=[figi])
                if not last_prices_response.last_prices:
                    continue

                price_quotation = last_prices_response.last_prices[0].price
                curr_price = quotation_to_float(price_quotation)

                instr_info = client.instruments.find_instrument(query=figi)
                name = "Unknown"
                ticker = "N/A"
                if instr_info.instruments:
                    instr = instr_info.instruments[0]
                    name = instr.name
                    ticker = instr.ticker

                value = balance * curr_price
                total_value_portfolio += value
                pl = value - (balance * avg) if avg > 0 else value  # P/L, если ср.цена есть

                items.append({
                    "figi": figi,
                    "ticker": ticker,
                    "name": name,
                    "quantity": balance,  # На фронте используется quantity
                    "avg_price": avg,
                    "curr_price": curr_price,
                    "value": value,
                    "pl": pl
                })

            # total_value можно взять из port.total_amount_portfolio или посчитать сумму items
            # Используем рассчитанную сумму для текущего стрима
            payload = {"type": "initial", "data": {"positions": items, "total": total_value_portfolio}}
            yield f"data: {json.dumps(payload)}\n\n"
            break  # Отправляем только initial и завершаем генератор, т.к. нет обработки обновлений


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/portfolio/stream")
def stream_portfolio():
    try:
        return StreamingResponse(
            portfolio_generator(API_TOKEN, ACCOUNT_ID),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error in stream_portfolio endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error streaming portfolio: {str(e)}")


@app.get("/candles")
def get_candles(figi: str, days: int = 30):
    try:
        with Client(API_TOKEN) as client:
            cache = MarketDataCache(
                settings=MarketDataCacheSettings(base_cache_dir=Path("cache")),
                services=client
            )
            start_date = now() - timedelta(days=days)

            candles_data = list(cache.get_all_candles(
                figi=figi,
                from_=start_date,
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))

            if not candles_data:
                return JSONResponse([])

            response_candles = []
            for c in candles_data:
                if c.time and c.close is not None:
                    response_candles.append({
                        "time": c.time.isoformat(),
                        "open": quotation_to_float(c.open),
                        "close": quotation_to_float(c.close)
                    })

            return JSONResponse(response_candles)
    except Exception as e:
        print(f"Error in get_candles for FIGI {figi}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error processing candles for {figi}: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)