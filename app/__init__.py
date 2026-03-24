from flask import Flask
from . import pageRoutes3
from .events import socketio

def create_app():
    app = Flask(__name__)
    app.config["DEBUG"] = True
    app.config["SECRET_KEY"] = "secret"

    app.register_blueprint(pageRoutes3.bp)
    socketio.init_app(app, async_mode='threading')

    return app



# The idea here is for cleaner setup for pages
# somewhere in the pages probably i want to setup sockets
# this is only for initialization

# run "flask --app __init__ run"

