import datetime as dt


class Timer:
    # A simple timer class to measure elapsed time.

    def __init__(self):
        self.start_dt = None

    def start(self):
        # Start the timer
        self.start_dt = dt.datetime.now()

    def stop(self):
        # Stop the timer and print the elapsed time
        if self.start_dt is None:
            raise ValueError("Timer was not started. Call start() before stop().")
        
        end_dt = dt.datetime.now()
        elapsed_time = end_dt - self.start_dt
        print(f"Time Taken: {elapsed_time}")
