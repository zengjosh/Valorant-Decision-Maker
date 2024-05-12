from pynput import keyboard
from PIL import Image
import pyautogui
import datetime
import os

# Define the hotkey and the kill switch combination
HOTKEY = '='
KILL_SWITCH = '`'

# The currently active modifiers
current_keys = set()

def get_latest_screenshot():
    screenshot_dir = "screenshots"
    # Ensure the directory exists and has files
    if os.path.exists(screenshot_dir) and os.listdir(screenshot_dir):
        # List all files in the directory
        files = [os.path.join(screenshot_dir, f) for f in os.listdir(screenshot_dir)]
        # Sort files by modification time in descending order
        latest_file = max(files, key=os.path.getmtime)
        print(f"The most recent screenshot is: {latest_file}")
        return latest_file
    else:
        print("No screenshots found.")
        return None

def take_screenshot():
    # Directory where the screenshots will be saved
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)  # Ensure the directory exists

    # Get the list of files in the screenshot directory
    files = os.listdir(screenshot_dir)
    files = [os.path.join(screenshot_dir, f) for f in files]  # Full paths of files
    files.sort(key=os.path.getmtime)  # Sort files by modification time, oldest first

    # # If there is more than one file in the directory, delete the oldest
    # print(f"Number of screenshots in directory: {len(files)}")
    # if len(files) >= 1:
    #     os.remove(files[0])
    #     print(f"Deleted oldest screenshot: {files[0]}")

    # Continue with saving the new screenshot
    filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(screenshot_dir, filename)
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    print(f"Screenshot taken and saved as {path}")

def on_press(key):
    # Add the key to the set of currently pressed keys
    if key == keyboard.KeyCode.from_char(HOTKEY):
        take_screenshot()
    elif key == keyboard.KeyCode.from_char(KILL_SWITCH):
        return False

def on_release(key):
    # Remove the key from the set of currently pressed keys
    try:
        current_keys.remove(key)
    except KeyError:
        pass  # Key was not in the set, ignore

# Set up the listener for keyboard events
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


# The script will stop here if the listener has stopped (e.g., after Ctrl+C is pressed)
print("Listener stopped. Program ended.")
