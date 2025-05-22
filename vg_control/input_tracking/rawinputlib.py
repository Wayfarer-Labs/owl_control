from typing import Tuple

import ctypes
from ctypes import Structure, c_ulonglong, c_long, c_uint, c_bool

import os
import time
import asyncio

from .keybinds import CODE_TO_KEY

class MouseMoveData(Structure):
    _fields_ = [("timestamp", c_ulonglong),
                ("dx", c_long),
                ("dy", c_long)]

class MouseButtonData(Structure):
    _fields_ = [("timestamp", c_ulonglong),
                ("button", c_uint),
                ("down", c_bool)]

class MouseScrollData(Structure):
    _fields_ = [("timestamp", c_ulonglong),
                ("scrollAmount", c_long)]

class KeyboardData(Structure):
    _fields_ = [("timestamp", c_ulonglong),
                ("keyCode", c_uint),
                ("down", c_bool)]

def keycode_to_key(key: int):
    try:
        return chr(key)
    except ValueError:
        return None

class RawInputReader:
    def __init__(self, dll_path = "./rawinputlib.dll"):
        self.dll_path = dll_path
        self.rawinput_lib = None
        self._load_dll()

        self.rawinput_lib.initialize_raw_input.restype = ctypes.c_int
        self.rawinput_lib.get_mouse_move_input.argtypes = [ctypes.POINTER(MouseMoveData)]
        self.rawinput_lib.get_mouse_move_input.restype = ctypes.c_int
        self.rawinput_lib.get_mouse_button_input.argtypes = [ctypes.POINTER(MouseButtonData)]
        self.rawinput_lib.get_mouse_button_input.restype = ctypes.c_int
        self.rawinput_lib.get_mouse_scroll_input.argtypes = [ctypes.POINTER(MouseScrollData)]
        self.rawinput_lib.get_mouse_scroll_input.restype = ctypes.c_int
        self.rawinput_lib.get_keyboard_input.argtypes = [ctypes.POINTER(KeyboardData)]
        self.rawinput_lib.get_keyboard_input.restype = ctypes.c_int
        self.rawinput_lib.cleanup_raw_input.restype = None

        self.mouse_move_data = MouseMoveData()
        self.mouse_button_data = MouseButtonData()
        self.mouse_scroll_data = MouseScrollData()
        self.keyboard_data = KeyboardData()

    def _load_dll(self):
        import os

        try:
            self.rawinput_lib = ctypes.CDLL(self.dll_path)
        except OSError:
            for root, dirs, files in os.walk('.'):
                if 'rawinputlib.dll' in files:
                    dll_path = os.path.join(root, 'rawinputlib.dll')
                    try:
                        self.rawinput_lib = ctypes.CDLL(dll_path)
                        self.dll_path = dll_path
                        break
                    except OSError:
                        continue

        if self.rawinput_lib is None:
            raise FileNotFoundError(f"Could not find or load rawinputlib dll.")

    def open(self) -> bool:
        """
        Get ready to read raw input, starts C process
        Returns bool indicating success
        """
        return self.rawinput_lib.initialize_raw_input()

    
    def close(self):
        """
        Close reading process and release handlers
        """
        self.rawinput_lib.cleanup_raw_input()

    def get_mouse_move_input(self) -> Tuple[bool, Tuple]:
        """
        Get mouse movements if any were polled
        Returns True and results if something was detected,
        Returns False, and an empty tuple otherwise
        """
        if self.rawinput_lib.get_mouse_move_input(ctypes.byref(self.mouse_move_data)):
            return True, (self.mouse_move_data.timestamp, self.mouse_move_data.dx, self.mouse_move_data.dy)
        else:
            return False, ()

    def get_mouse_button_input(self) -> Tuple[bool, Tuple]:
        """
        Get mouse button events if any were polled
        Returns True and results if something was detected,
        Returns False, and an empty tuple otherwise
        """
        if self.rawinput_lib.get_mouse_button_input(ctypes.byref(self.mouse_button_data)):
            return True, (self.mouse_button_data.timestamp, self.mouse_button_data.button, self.mouse_button_data.down)
        else:
            return False, ()

    def get_mouse_scroll_input(self) -> Tuple[bool, Tuple]:
        """
        Get mouse scroll events if any were polled
        Returns True and results if something was detected,
        Returns False, and an empty tuple otherwise
        """
        if self.rawinput_lib.get_mouse_scroll_input(ctypes.byref(self.mouse_scroll_data)):
            return True, (self.mouse_scroll_data.timestamp, self.mouse_scroll_data.scrollAmount)
        else:
            return False, ()

    def get_keyboard_input(self) -> Tuple[bool, Tuple]:
        """
        Get keyboard events if any were polled
        Returns True and results if something was detected,
        Returns False, and an empty tuple otherwise
        """
        if self.rawinput_lib.get_keyboard_input(ctypes.byref(self.keyboard_data)):
            return True, (
                self.keyboard_data.timestamp,
                self.keyboard_data.keyCode,
                self.keyboard_data.down
            )
        else:
            return False, ()

from ..constants import POLLS_PER_FRAME, FPS

RAW_INPUT = RawInputReader()
if not RAW_INPUT.open():
    raise RuntimeError("Failed to initialize raw input")

async def wait_until_input():
    """
    Blocks until any input is detected then 
    """
    delay = 0.05
    reader = RAW_INPUT

    try:
        while True:
            mouse_move_success, _ = reader.get_mouse_move_input()
            if mouse_move_success:
                return

            mouse_button_success, _ = reader.get_mouse_button_input() 
            if mouse_button_success:
                return

            mouse_scroll_success, _ = reader.get_mouse_scroll_input()
            if mouse_scroll_success:
                return

            keyboard_success, _ = reader.get_keyboard_input()
            if keyboard_success:
                return

            await asyncio.sleep(delay)
    except:
        return
    
async def wait_until_idle(max_idle_time):
    delay = 0.05
    idle_counter = 0
    reader = RAW_INPUT
    last_time = time.perf_counter()

    try:
        while True:
            mouse_move_success, _ = reader.get_mouse_move_input()
            if mouse_move_success:
                idle_counter = 0

            mouse_button_success, _ = reader.get_mouse_button_input() 
            if mouse_button_success:
                idle_counter = 0

            mouse_scroll_success, _ = reader.get_mouse_scroll_input()
            if mouse_scroll_success:
                idle_counter = 0

            keyboard_success, _ = reader.get_keyboard_input()
            if keyboard_success:
                idle_counter = 0

            await asyncio.sleep(delay)
            time_delta = time.perf_counter() - last_time
            last_time = time.perf_counter()
            idle_counter += time_delta

            if idle_counter > max_idle_time:
                return
    except asyncio.CancelledError:
        raise
    except:
        return

class HotkeyManager:
    def __init__(self):
        self.reader = RAW_INPUT
        self.callbacks = {}
        self.delay = 1. / FPS / POLLS_PER_FRAME

        self.tasks = set()

    async def event_loop(self):
        while True:
            await asyncio.sleep(self.delay)
            kb_success, kb_info = self.reader.get_keyboard_input()
            if not kb_success: # Nothing pressed
                continue
            keydown = kb_info[2]
            if not keydown: # Not a down event
                continue
            keycode = kb_info[1]
            if keycode in self.callbacks:
                # Create a new task instead of awaiting directly
                await self.call(keycode)
                #task = asyncio.create_task(self.call(keycode))
                #self.tasks.add(task)
                #task.add_done_callback(self.tasks.discard)

    async def call(self, keycode):
        await self.callbacks[keycode]()

    def add_callback(self, keycode, fn):
        if not isinstance(keycode, int):
            keycode = keycode.upper()
            try:
                keycode = {k for k,v in CODE_TO_KEY.items() if v == keycode}.pop()
            except:
                raise ValueError("Key {keycode} not supported by hotkey manager.")
        self.callbacks[keycode] = fn