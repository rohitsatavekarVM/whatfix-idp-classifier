"""
Application entry point.
"""

from quart import Quart, jsonify

from app.routes.whatfix_routes import whatfix_bp


def create_app() -> Quart:
    app = Quart(__name__)

    # Register blueprints
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
        return jsonify(
            {
                "status": "UP"
            }
        )

    @app.get("/routes")
    async def routes():
        routes = []

        for rule in app.url_map.iter_rules():
            routes.append(
                {
                    "endpoint": rule.endpoint,
                    "methods": sorted(
                        m for m in rule.methods
                        if m not in ("HEAD", "OPTIONS")
                    ),
                    "path": str(rule),
                }
            )

        return jsonify(routes)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )