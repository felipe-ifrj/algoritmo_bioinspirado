class Application:

    def __init__(self, required_sensors):
        self.required_sensors = required_sensors

    def __str__(self):
        return str (self.required_sensors)
    