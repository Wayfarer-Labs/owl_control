import time
import asyncio

from .rawinputlib import RawInputReader
from .spam_block import SpamBlock
from ..constants import FPS, POLLS_PER_FRAME

# This object uses RawInputReader in a loop
# to continuously collect input
# note that we cut out timestamp as its preferable to just get timestamp in python
class InputTracker:
    def __init__(self):
        self.raw_input_reader = RawInputReader()
        self.running = False
        self.callbacks = {}
        self.polling_delay = 1. / (FPS * POLLS_PER_FRAME)

        self.kb_spam_blocker = None
        self.mb_spam_blocker = None

        self.last_event_time = time.perf_counter()

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def get_time_since_last_event(self):
        """Returns the time in seconds since the last input event was received."""
        return time.perf_counter() - self.last_event_time


    async def __call__(self):
        if not self.raw_input_reader.open():
            raise RuntimeError("Failed to initialize raw input")

        self.running = True
        try:
            self.kb_spam_blocker = SpamBlock()
            self.mb_spam_blocker = SpamBlock()
            self.callbacks['keyboard'] = self.kb_spam_blocker.decorate(self.callbacks['keyboard'])
            self.callbacks['mouse_button'] = self.mb_spam_blocker.decorate(self.callbacks['mouse_button'])
            while self.running:
                any_event = False
                mouse_move_success, mouse_move_data = self.raw_input_reader.get_mouse_move_input()
                if mouse_move_success:
                    self.callbacks['mouse_move'](*mouse_move_data[1:])
                    any_event = True
                
                mouse_button_success, mouse_button_data = self.raw_input_reader.get_mouse_button_input()
                if mouse_button_success:
                    self.callbacks['mouse_button'](*mouse_button_data[1:])
                    any_event = True
                
                mouse_scroll_success, mouse_scroll_data = self.raw_input_reader.get_mouse_scroll_input()
                if mouse_scroll_success:
                    self.callbacks['mouse_scroll'](*mouse_scroll_data[1:])
                    any_event = True
                
                keyboard_success, keyboard_data = self.raw_input_reader.get_keyboard_input()
                if keyboard_success:
                    self.callbacks['keyboard'](*keyboard_data[1:])
                    any_event = True

                if any_event:
                    self.last_event_time = time.perf_counter()

                await asyncio.sleep(self.polling_delay)  # Small sleep to reduce CPU usage
        finally:
            self.raw_input_reader.close()
            self.kb_spam_blocker = None
            self.mb_spam_blocker = None

    def stop(self):
        self.running = False

if __name__ == "__main__":
    import asyncio

    async def main():
        tracker = InputTracker()
        
        def mouse_move_callback(dx, dy):
            timestamp = time.perf_counter()
            print(f"Mouse Move: timestamp={timestamp}, dx={dx}, dy={dy}")
        
        def mouse_button_callback(button, down):
            timestamp = time.perf_counter()
            print(f"Mouse Button: timestamp={timestamp}, button={button}, down={down}")
        
        def mouse_scroll_callback(scroll_amount):
            timestamp = time.perf_counter()
            print(f"Mouse Scroll: timestamp={timestamp}, scroll_amount={scroll_amount}")
        
        def keyboard_callback(key_code, down):
            timestamp = time.perf_counter()
            print(f"Keyboard: timestamp={timestamp}, key_code={key_code}, down={down}")
        callbacks = {
            'mouse_move': mouse_move_callback,
            'mouse_button': mouse_button_callback,
            'mouse_scroll': mouse_scroll_callback,
            'keyboard': keyboard_callback
        }
        
        tracker.set_callbacks(callbacks)
        
        task = asyncio.create_task(tracker())
        
        try:
            await asyncio.sleep(10)  # Run for 10 seconds
        finally:
            tracker.stop()
            await task

    asyncio.run(main())


