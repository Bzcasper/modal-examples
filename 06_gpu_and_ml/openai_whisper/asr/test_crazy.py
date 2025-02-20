from modal import App, asgi_app, Secret, Image
import time

app = App("latency_test_crazy_final")


@app.function(keep_warm=1, secrets=[Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})], image = Image.debian_slim().pip_install("fastapi"))
@asgi_app()
def web():
    from fastapi import FastAPI, Request
    web_app = FastAPI()

    @web_app.post("/")
    async def predict(request: Request):
        print(f"Start reading at {time.monotonic()}")
        data = await request.body()
        print(f"End reading at {time.monotonic()}")
    
    return web_app
