import argparse
import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, jsonify

from app.routes.whatfix_routes import whatfix_bp


app = Quart(__name__)
app.register_blueprint(whatfix_bp)



@app.get("/")
async def home():
    return jsonify(
        {
            "service": "Whatfix Migration Service",
            "version": "1.0.0",
            "status": "running",
        }
    )


@app.get("/health")
async def health():
    return jsonify({"status": "UP"})

print("\n===== REGISTERED ROUTES =====")

for rule in app.url_map.iter_rules():
    print(rule)

print("=============================\n")


async def shutdown_trigger():
    await asyncio.Event().wait()


async def main(port: int):

    config = Config()

    config.bind = [f"0.0.0.0:{port}"]
    config.accesslog = "-"
    config.use_reloader = False

    await serve(
        app,
        config=config,
        shutdown_trigger=shutdown_trigger,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        default=11001,
    )

    args = parser.parse_args()

    asyncio.run(main(args.port))