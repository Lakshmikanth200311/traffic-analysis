import time

TRAFFIC_STATES = ['RED', 'GREEN', 'YELLOW']
DEFAULT_TIMINGS = {
    'RED': 10,
    'GREEN': 10,
    'YELLOW': 3
}

# simulate adaptive light control based on vehicle count

def simulate_traffic_logic(frame_data):
    # simulate vehicle detection here (mock for now)
    vehicle_count = frame_data.get('vehicle_count', 0)
    timings = DEFAULT_TIMINGS.copy()

    if vehicle_count > 10:
        timings['RED'] += 5  # increase red light if lane is congested

    return {
        'status': 'ok',
        'vehicle_count': vehicle_count,
        'current_light': 'RED',
        'timings': timings
    }
