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
ACCOUNT_ID = "2039833462"

app = FastAPI(title="Tinkoff Portfolio Stream API")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def index():
    return "static/portfolio.html"

def portfolio_generator(token: str, account_id: str):
    """SSE: отдаём сразу всю корзину со всеми нужными полями."""
    with Client(token) as client:
        # получаем портфель для avg_price
        port = client.operations.get_portfolio(account_id=account_id)
        avg_price_map = {}
        for p in port.positions:
            try:
                avg_price_map[p.figi] = float(p.average_position_price.value or 0)
            except Exception:
                continue

        # получаем список аккаунтов для стрима
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

                # получаем последнюю цену
                last = client.market_data.get_last_prices(figi=[figi]).last_prices[0].price
                curr_price = float(last.units) + last.nano / 1e9
                pos_value = balance * curr_price
                total_value += pos_value

                # ищем инструмент по FIGI
                resp_instr = client.market.market_search_by_figi_get(figi)
                payload = resp_instr.payload
                name = payload['name']
                ticker = payload['ticker']

                items.append({
                    "figi": figi,
                    "ticker": ticker,
                    "name": name,
                    "quantity": balance,
                    "avg_price": avg,
                    "curr_price": curr_price,
                    "value": pos_value,
                    "pl": pos_value - balance * avg
                })

            payload = {"type": "initial", "data": {"positions": items, "total": total_value}}
            yield f"data: {json.dumps(payload)}\n\n"


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
                figi=figi, from_=start, interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))
            return JSONResponse([{
                "time": c.time.isoformat(),
                "open": float(c.open),
                "close": float(c.close)
            } for c in candles])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))