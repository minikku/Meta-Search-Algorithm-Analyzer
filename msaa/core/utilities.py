from datetime import datetime
import time

def show_current_date_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def show_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    hours, minutes, seconds = int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60)
    return f"{hours} hours, {minutes} minutes, and {seconds} seconds"
