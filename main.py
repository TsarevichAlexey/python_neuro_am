
from config import TOKEN
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tinkoff.invest import Client
import uvicorn
import json

from fastapi.staticfiles import StaticFiles

# Замените на ваш персональный токен от Tinkoff Invest API
API_TOKEN = TOKEN

# Если у вас несколько счетов, вы можете указать нужный
# либо получать список автоматически через Client.users.get_accounts()
ACCOUNT_ID = "2039833462"  # или строка с ID вашего счета


app = FastAPI(title="Tinkoff Portfolio Stream API")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
class MoneyPosition(BaseModel):
    currency: str
    units: int
    nano: int

class SecurityPosition(BaseModel):
    figi: str
    blocked: float
    balance: float
    position_uid: str

class PortfolioResponse(BaseModel):
    money: list[MoneyPosition]
    securities: list[SecurityPosition]


def portfolio_generator(token: str, account_id: str):
    """Генератор событий для SSE: шлет начальные позиции и обновления в реальном времени."""
    with Client(token) as client:
        # Получаем список аккаунтов
        resp = client.users.get_accounts()
        accounts = [acc.id for acc in resp.accounts]
        if account_id not in accounts:
            accounts = [account_id]

        stream = client.operations_stream.positions_stream(
            accounts=accounts,
            with_initial_positions=True
        )
        # Проходим по каждому сообщению
        for update in stream:
            payload = None
            # Начальные позиции
            if update.initial_positions:
                init = update.initial_positions
                money = [m.dict() for m in [MoneyPosition(currency=m.currency, units=m.units, nano=m.nano) for m in init.money]]
                securities = [s.dict() for s in [SecurityPosition(figi=s.figi, blocked=s.blocked, balance=s.balance, position_uid=s.position_uid) for s in init.securities]]
                payload = {"type": "initial", "data": {"money": money, "securities": securities}}
            # Обновление отдельной позиции
            elif update.position:
                p = update.position
                payload = {"type": "update", "data": {"figi": p.figi, "balance": p.balance, "blocked": p.blocked, "position_uid": p.position_uid}}
            # Пинг или другое сообщение - пропускаем
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"

@app.get("/portfolio/stream")
def stream_portfolio():
    """Endpoint SSE для реального времени изменений портфеля."""
    try:
        return StreamingResponse(
            portfolio_generator(API_TOKEN, ACCOUNT_ID),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при стриминге портфеля: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
