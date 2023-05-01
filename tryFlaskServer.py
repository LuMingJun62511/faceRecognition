import socket
import eventlet.wsgi
from flask import Flask, render_template
from flask_socketio import SocketIO, Namespace, emit, join_room, leave_room

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

class MyNamespace(Namespace):
    def on_connect(self):
        print('Client connected', self.namespace)

    def on_disconnect(self):
        print('Client disconnected', self.namespace)

    def on_join(self, room):
        join_room(room)
        emit('status', {'msg': 'Client joined room: ' + room}, room=room)

    def on_leave(self, room):
        leave_room(room)
        emit('status', {'msg': 'Client left room: ' + room}, room=room)

    def on_message(self, data):
        emit('message', data, room=data['room'])

socketio.on_namespace(MyNamespace('/my_namespace'))

@app.route('/')
def index():
    return render_template('frontend.html')

if __name__ == '__main__':
    print("Server started connect with http://127.0.0.1:8080 or http://"+IPAddr+":8080")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8080)), app)