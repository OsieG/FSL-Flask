from .extensions import socketio

#   handles the server requests
#   this helps with loading the events/listeners onto "socketio"


@socketio.on("connect")
def handle_connect():        #remember this needs to be snake_case
    print("client connected!")