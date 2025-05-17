
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tinkoff.invest import Client, CandleInterval, Quotation, MoneyValue
from tinkoff.invest.utils import now

import subprocess
import pandas as pd
from datetime import timedelta
from pathlib import Path
import json
import traceback
import os

from config import TOKEN

import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
     print("Note: Could not reconfigure stdout/stderr buffering.")


API_TOKEN = TOKEN
ACCOUNT_ID = "2246688089"

if not ACCOUNT_ID or ACCOUNT_ID == "YOUR_ACCOUNT_ID":
    print("WARNING: ACCOUNT_ID is not set in main.py. Portfolio features will likely fail.")


app = FastAPI(title="Tinkoff Portfolio Stream API")

static_dir = Path(__file__).parent / "static"
if not static_dir.is_dir():
    print(f"WARNING: Static directory '{static_dir}' not found. Frontend may not load.")
else:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def quotation_to_float(q: Quotation) -> float:
    if q is None: return 0.0
    return q.units + q.nano / 1_000_000_000.0

def money_value_to_float(mv: MoneyValue) -> float:
    if mv is None: return 0.0
    return mv.units + mv.nano / 1_000_000_000.0

@app.get("/", response_class=FileResponse)
def index():
    html_path = static_dir / "portfolio.html"
    if not html_path.is_file():
         print(f"ERROR: portfolio.html not found at {html_path}")
         raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(html_path)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = static_dir / "favicon.ico"
    if favicon_path.is_file():
        return FileResponse(favicon_path)
    else:
        return Response(status_code=204)


def portfolio_generator(token: str, account_id: str):
    print(f"Starting portfolio stream for account: {account_id}")
    with Client(token) as client:
        try:
            try:
                port = client.operations.get_portfolio(account_id=account_id)
                avg_price_map = {p_pos.figi: money_value_to_float(p_pos.average_position_price)
                                 for p_pos in port.positions}

                initial_items = []
                total_value_portfolio = 0.0
                figis_to_query = [p.figi for p in port.positions]

                current_prices = {}
                if figis_to_query:
                    try:
                        last_prices_response = client.market_data.get_last_prices(figi=figis_to_query)
                        current_prices = {lp.figi: quotation_to_float(lp.price)
                                          for lp in last_prices_response.last_prices}
                    except Exception as e_price:
                        print(f"Warning: Could not get batch prices: {e_price}")

                instrument_info = {}
                if figis_to_query:
                     try:
                         for figi in figis_to_query:
                             try:
                                 instr_resp = client.instruments.get_instrument_by(id_type=1, id=figi)
                                 if instr_resp and instr_resp.instrument:
                                     instr = instr_resp.instrument
                                     instrument_info[figi] = {'name': instr.name, 'ticker': instr.ticker}
                             except Exception as e_instr_single:
                                  print(f"Could not get instrument info for {figi}: {e_instr_single}")
                                  instrument_info[figi] = {'name': 'Unknown Name', 'ticker': 'N/A'}
                     except Exception as e_instr_batch:
                          print(f"Error during batch instrument info fetch (using loop): {e_instr_batch}")


                for p_pos in port.positions:
                    figi = p_pos.figi
                    balance_qnt = quotation_to_float(p_pos.quantity)
                    avg = avg_price_map.get(figi, 0.0)
                    curr_price = current_prices.get(figi, 0.0)
                    info = instrument_info.get(figi, {'name': 'Unknown Name', 'ticker': 'N/A'})

                    value = balance_qnt * curr_price
                    total_value_portfolio += value
                    pl = (curr_price - avg) * balance_qnt if avg > 0 else value


                    initial_items.append({
                        "figi": figi,
                        "ticker": info['ticker'],
                        "name": info['name'],
                        "quantity": balance_qnt,
                        "avg_price": avg,
                        "curr_price": curr_price,
                        "value": value,
                        "pl": pl
                    })

                payload = {"type": "initial", "data": {"positions": initial_items, "total": total_value_portfolio}}
                yield f"data: {json.dumps(payload)}\n\n"
                print("Initial portfolio snapshot sent.")

            except Exception as e_init:
                 print(f"Error getting initial portfolio: {e_init}")
                 traceback.print_exc()
                 error_payload = {"type": "error", "message": f"Could not load initial portfolio: {str(e_init)}"}
                 yield f"data: {json.dumps(error_payload)}\n\n"

        except Exception as e:
            print(f"Error in portfolio_generator: {e}")
            traceback.print_exc()
            error_payload = {"type": "error", "message": f"Server error in portfolio stream: {str(e)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"

    print(f"Portfolio stream stopped for account: {account_id}")


@app.get("/portfolio/stream")
def stream_portfolio():
    if not API_TOKEN:
        print("Error: API_TOKEN not configured.")
        raise HTTPException(status_code=500, detail="API_TOKEN not configured on server.")
    if not ACCOUNT_ID or ACCOUNT_ID == "YOUR_ACCOUNT_ID":
        print("Error: ACCOUNT_ID not configured.")
        raise HTTPException(status_code=500, detail="ACCOUNT_ID not configured on server.")

    print(f"Client requested portfolio stream for account {ACCOUNT_ID}")
    return StreamingResponse(
        portfolio_generator(API_TOKEN, ACCOUNT_ID),
        media_type="text/event-stream",
        headers={'Cache-Control': 'no-cache'}
    )

@app.get("/candles")
def get_candles(figi: str, days: int = 30):
    print(f"Received candles request for FIGI: {figi}, days: {days}")
    if not figi:
        raise HTTPException(status_code=400, detail="FIGI parameter is required.")
    if not API_TOKEN:
         print("Error: API_TOKEN not configured for /candles endpoint.")
         raise HTTPException(status_code=500, detail="API_TOKEN not configured on server.")

    try:
        with Client(API_TOKEN) as client:
            start_date = now() - timedelta(days=days)
            end_date = now()

            candles_data = list(client.get_all_candles(
                figi=figi,
                from_=start_date,
                to=end_date,
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            ))

            if not candles_data:
                print(f"No candle data found for FIGI {figi} in the last {days} days.")
                return JSONResponse(content=[])

            response_candles = []
            for c in candles_data:
                if c.time and c.close and c.open and c.high and c.low:
                    response_candles.append({
                        "time": c.time.isoformat(),
                        "open": quotation_to_float(c.open),
                        "close": quotation_to_float(c.close),
                        "high": quotation_to_float(c.high),
                        "low": quotation_to_float(c.low),
                        "volume": c.volume
                    })
            print(f"Returning {len(response_candles)} candles for FIGI {figi}.")
            return JSONResponse(content=response_candles)

    except Exception as e:
        print(f"Error in get_candles for FIGI {figi}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error processing candles for {figi}: {str(e)}")

@app.get("/forecast")
def forecast(figi: str = Query(..., description="FIGI инструмента для прогнозирования")):
    print(f"Received forecast request for FIGI: {figi}")
    if not figi:
        print("Forecast request failed: FIGI parameter missing.")
        raise HTTPException(status_code=400, detail="FIGI parameter is required.")
    if not API_TOKEN:
        print("Error: API_TOKEN not configured for /forecast endpoint.")
        raise HTTPException(status_code=500, detail="API_TOKEN not configured on server.")


    base_dir = Path(__file__).parent
    script_path = base_dir / "model2.py"
    forecast_file_path = base_dir / "forecast_result.csv"

    print(f"Script path: {script_path}")
    print(f"Forecast output file path: {forecast_file_path}")


    try:
        if not script_path.is_file():
            print(f"Error: Model script '{script_path}' not found.")
            raise HTTPException(status_code=500, detail=f"Model training script '{script_path.name}' not found on server.")

        if forecast_file_path.exists():
            try:
                forecast_file_path.unlink()
                print(f"Removed existing forecast file: {forecast_file_path}")
            except OSError as e:
                 print(f"Warning: Could not remove existing forecast file '{forecast_file_path}': {e}")

        python_executable = sys.executable
        command = [python_executable, str(script_path), figi]
        print(f"Running command: {' '.join(command)}")

        env = os.environ.copy()
        env["TINKOFF_API_TOKEN"] = API_TOKEN

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
            cwd=base_dir,
            env=env
        )

        print(f"--- Subprocess STDOUT (FIGI: {figi}) ---")
        print(process.stdout.strip() if process.stdout else "<empty stdout>")
        print(f"--- Subprocess STDERR (FIGI: {figi}) ---")
        print(process.stderr.strip() if process.stderr else "<empty stderr>")
        print(f"--- Subprocess Return Code (FIGI: {figi}): {process.returncode} ---")

        if process.returncode != 0:
            error_detail = f"Model script failed for {figi} with exit code {process.returncode}."
            if process.stderr:
                stderr_snippet = process.stderr.strip()[:500]
                error_detail += f" Stderr snippet: {stderr_snippet}"
            print(f"Error: Model script failed for FIGI {figi}. Return code: {process.returncode}")
            raise HTTPException(status_code=500, detail=error_detail)

        if not forecast_file_path.is_file():
            print(f"Error: Forecast file '{forecast_file_path}' not found after script execution for FIGI {figi}.")
            raise HTTPException(status_code=500,
                                detail=f"Forecast result file not found for {figi} after script execution. Check server logs.")

        print(f"Reading forecast data from: {forecast_file_path}")
        try:
            df = pd.read_csv(forecast_file_path, index_col=0, parse_dates=True)
        except Exception as e_read:
             print(f"Error reading forecast CSV file '{forecast_file_path}': {e_read}")
             raise HTTPException(status_code=500, detail=f"Failed to read forecast result file for {figi}.")


        if "predicted_close" not in df.columns:
            print(f"Error: 'predicted_close' column not found in {forecast_file_path} for FIGI {figi}.")
            raise HTTPException(status_code=500,
                                detail=f"Forecast data format error for {figi}: 'predicted_close' column missing.")

        if df.empty:
            print(f"Forecast data file is empty for FIGI {figi}. Returning empty forecast.")
            times_iso = []
            prices = []
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index in {forecast_file_path} is not datetime. Attempting conversion.")
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e_date:
                    print(f"Error converting index to datetime: {e_date}")
                    raise HTTPException(status_code=500, detail=f"Forecast data format error for {figi}: Cannot parse index as dates.")

            df = df.sort_index()
            times_iso = [ts.isoformat() for ts in df.index]
            prices = df["predicted_close"].tolist()

        data = {
            "times": times_iso,
            "prices": prices
        }
        print(f"Successfully generated forecast for {figi} with {len(prices)} points.")
        return JSONResponse(content=data)

    except subprocess.TimeoutExpired:
        print(f"Timeout error running model script for FIGI {figi}")
        traceback.print_exc()
        raise HTTPException(status_code=508, detail=f"Model script timed out for {figi} (limit: 600 seconds).")
    except FileNotFoundError as e:
        print(f"Script execution error (FileNotFoundError): {e}")
        traceback.print_exc()
        detail = f"Model script execution error: {e.strerror}."
        if e.filename:
            detail += f" File not found: {e.filename}"
        raise HTTPException(status_code=500, detail=detail)
    except HTTPException:
         raise
    except Exception as e:
        print(f"Unhandled error in forecast endpoint for FIGI {figi}:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error processing forecast for {figi}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
