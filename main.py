from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.caching.market_data_cache.cache import MarketDataCache
from tinkoff.invest.caching.market_data_cache.cache_settings import MarketDataCacheSettings
from tinkoff.invest.utils import now
from pydantic import BaseModel
from datetime import timedelta
from pathlib import Path
import os, json

from config import TOKEN
API_TOKEN = TOKEN
ACCOUNT_ID = "2246688089"

app = FastAPI(title="Tinkoff Portfolio Stream API")
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=FileResponse)
def index():
    return "static/portfolio.html"


def portfolio_generator(token: str, account_id: str):
    """SSE: отдаём сразу всю корзину со всеми нужными полями."""
    with Client(token) as client:
        # Получаем avg цены из портфеля
        port = client.operations.get_portfolio(account_id=account_id)
        avg_price_map = {}
        for p in port.positions:
            try:
                avg_price_map[p.figi] = float(p.average_position_price.value or 0)
            except AttributeError:
                continue

        # Подписка
        resp = client.users.get_accounts()
        accounts = [a.id for a in resp.accounts]
        if account_id not in accounts:
            accounts = [account_id]

        stream = client.operations_stream.positions_stream(
            accounts=accounts, with_initial_positions=True
        )
        for update in stream:
            if not update.initial_positions:
                continue
            init = update.initial_positions
            items = []
            total_value = 0.0
            for s in init.securities:
                figi = s.figi
                balance = s.balance
                avg = avg_price_map.get(figi, 0.0)

                # Текущая цена
                last_prices = client.market_data.get_last_prices(figi=[figi]).last_prices
                if not last_prices:
                    continue
                price = last_prices[0].price
                curr_price = float(price.units) + price.nano / 1e9

                # Информация об инструменте через find_instrument
                instr_list = client.instruments.find_instrument(query=figi).instruments
                if instr_list:
                    instr = instr_list[0]  # InstrumentShort
                    name = instr.name  # напрямую у InstrumentShort
                    ticker = instr.ticker
                else:
                    name = "Unknown"
                    ticker = "N/A"

                value = balance * curr_price
                total_value += value
                pl = value - balance * avg

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

            payload = {"type": "initial", "data": {"positions": items, "total": total_value}}
            yield f"data: {json.dumps(payload)}\n\n"


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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candles")
def get_candles(figi: str, days: int = 30):
    try:
        with Client(API_TOKEN) as client:
            cache = MarketDataCache(
                settings=MarketDataCacheSettings(base_cache_dir=Path("cache")),
                services=client
            )
            start = now() - timedelta(days=days)
            candles = list(cache.get_all_candles(
                figi=figi,
                from_=start,
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))
            return JSONResponse([
                {"time": c.time.isoformat(), "open": float(c.open), "close": float(c.close)}
                for c in candles
            ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
