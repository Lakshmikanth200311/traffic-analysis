import time
from threading import Thread, Lock
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Configuration ---
ROUTES = ['north', 'east', 'west', 'south']
DEFAULT_GREEN = 30
DEFAULT_YELLOW = 15
DEFAULT_RED = 60

# Shared data
vehicle_counts = {route: 0 for route in ROUTES}
cycle_timings = {route: DEFAULT_GREEN for route in ROUTES}
lock = Lock()

# Control flag for run loop
running_cycle = False

def calculate_cycle_timings(counts):
    # Simple logic: proportionally more green time for more vehicles
    total = sum(counts.values())
    timings = {}
    for route in ROUTES:
        if total == 0:
            timings[route] = DEFAULT_GREEN
        else:
            # Example: allocate green based on vehicle ratio (cap min/max)
            green = int(DEFAULT_GREEN + (counts[route]/total) * 30)
            timings[route] = max(15, min(green, 60))
    return timings

def run_cycle():
    global running_cycle
    while True:
        lock.acquire()
        current_timings = cycle_timings.copy()
        lock.release()
        for route in ROUTES:
            print(f"{route.capitalize()} - GREEN for {current_timings[route]} sec")
            time.sleep(current_timings[route])
            print(f"{route.capitalize()} - YELLOW for {DEFAULT_YELLOW} sec")
            time.sleep(DEFAULT_YELLOW)
            print(f"{route.capitalize()} - RED for {DEFAULT_RED} sec")
            time.sleep(DEFAULT_RED)
        # After every full cycle, recalculate timings using latest counts
        lock.acquire()
        cycle_timings.update(calculate_cycle_timings(vehicle_counts))
        lock.release()

@app.route('/update_counts', methods=['POST'])
def update_counts():
    global vehicle_counts, running_cycle
    data = request.json
    with lock:
        for route in ROUTES:
            if route in data:
                vehicle_counts[route] = int(data[route])
        cycle_timings.update(calculate_cycle_timings(vehicle_counts))
    return jsonify({'status': 'updated', 'cycle_timings': cycle_timings})

@app.route('/get_timings')
def get_timings():
    with lock:
        return jsonify(cycle_timings)

if __name__ == '__main__':
    # Start simulation loop in a background thread
    sim_thread = Thread(target=run_cycle, daemon=True)
    sim_thread.start()
    app.run(host='0.0.0.0', port=5000)
