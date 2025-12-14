import socketio

sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print("Connected! Sending counts.")
    sio.emit('update_counts', {'north': 45, 'south': 23, 'east': 5, 'west': 13})

@sio.on('cycle_plan')
def on_cycle_plan(data):
    print('Received cycle plan:', data)

sio.connect('http://localhost:6000')
sio.wait()
