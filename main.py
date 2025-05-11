from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.caching.market_data_cache.cache import MarketDataCache
from tinkoff.invest.caching.market_data_cache.cache_settings import MarketDataCacheSettings
from tinkoff.invest.utils import now
from pydantic import BaseModel
from pathlib import Path
import os, json
from config import TOKEN
API_TOKEN = TOKEN
ACCOUNT_ID = "2039833462"

app = FastAPI(title="Tinkoff Portfolio Stream API")

# 1) Монтируем папку static на /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) Отдаём HTML по корню
@app.get("/", response_class=FileResponse)
def index():
    return "static/portfolio.html"

# SSE-генератор
def portfolio_generator(token: str, account_id: str):
    with Client(token) as client:
        accounts = [acc.id for acc in client.users.get_accounts().accounts]
        if account_id not in accounts:
            accounts = [account_id]
        stream = client.operations_stream.positions_stream(
            accounts=accounts, with_initial_positions=True
        )
        for update in stream:
            payload = None
            if update.initial_positions:
                init = update.initial_positions
                securities = [
                    {
                        "figi": s.figi,
                        "balance": s.balance,
                        "blocked": s.blocked
                    }
                    for s in init.securities
                ]
                payload = {"type": "initial", "data": {"securities": securities}}
            elif update.position:
                p = update.position
                payload = {
                    "type": "update",
                    "data": {
                        "figi": p.figi,
                        "balance": p.balance,
                        "blocked": p.blocked
                    }
                }
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"

# 3) SSE-эндпоинт
@app.get("/portfolio/stream")
def stream_portfolio():
    try:
        return StreamingResponse(
            portfolio_generator(API_TOKEN, ACCOUNT_ID),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при стриминге портфеля: {e}")

# 4) Эндпоинт свечей
@app.get("/candles")
def get_candles(figi: str, days: int = 1):
    try:
        with Client(API_TOKEN) as client:
            cache_dir = Path("market_data_cache")
            settings = MarketDataCacheSettings(base_cache_dir=cache_dir)
            cache = MarketDataCache(settings=settings, services=client)
            start = now() - timedelta(days=days)
            candles = list(cache.get_all_candles(
                figi=figi, from_=start, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            ))
            return JSONResponse([
                {"time": c.time.isoformat(), "open": c.open, "close": c.close}
                for c in candles
            ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
