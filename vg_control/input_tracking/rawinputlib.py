from dataclasses import dataclass
from typing import Optional, Tuple, Union

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

class RawInputLib:
    def __init__(self, dll_path = "./rawinputlib.dll"):
        self.dll_path = dll_path
        self._load_dll()
        self._set_types()

    def _load_dll(self):
        import os

        self.cdll = None
        try:
            self.cdll = ctypes.CDLL(self.dll_path)
        except OSError:
            for root, dirs, files in os.walk('.'):
                if 'rawinputlib.dll' in files:
                    dll_path = os.path.join(root, 'rawinputlib.dll')
                    try:
                        self.cdll = ctypes.CDLL(dll_path)
                        self.dll_path = dll_path
                        break
                    except OSError:
                        continue
        finally:
            if self.cdll is None:
                raise FileNotFoundError(f"Could not find or load rawinputlib dll.")

    def _set_types(self):
        self.cdll.initialize_raw_input.restype = ctypes.c_int
        self.cdll.get_mouse_move_input.argtypes = [ctypes.POINTER(MouseMoveData)]
        self.cdll.get_mouse_move_input.restype = ctypes.c_int
        self.cdll.get_mouse_button_input.argtypes = [ctypes.POINTER(MouseButtonData)]
        self.cdll.get_mouse_button_input.restype = ctypes.c_int
        self.cdll.get_mouse_scroll_input.argtypes = [ctypes.POINTER(MouseScrollData)]
        self.cdll.get_mouse_scroll_input.restype = ctypes.c_int
        self.cdll.get_keyboard_input.argtypes = [ctypes.POINTER(KeyboardData)]
        self.cdll.get_keyboard_input.restype = ctypes.c_int
        self.cdll.cleanup_raw_input.restype = None


class MouseMoveInputReader:
    def __init__(self, rawinput_lib: RawInputLib):
        self.rawinput_lib = rawinput_lib

    def get(self) -> Optional[Tuple]:
        data = MouseMoveData()
        if self.rawinput_lib.cdll.get_mouse_move_input(ctypes.byref(data)):
            return data.timestamp, data.dx, data.dy

class MouseButtonInputReader:
    def __init__(self, rawinput_lib: RawInputLib):
        self.rawinput_lib = rawinput_lib

    def get(self) -> Optional[Tuple]:
        data = MouseButtonData()
        if self.rawinput_lib.cdll.get_mouse_button_input(ctypes.byref(data)):
            return data.timestamp, data.button, data.down

class MouseScrollInputReader:
    def __init__(self, rawinput_lib: RawInputLib):
        self.rawinput_lib = rawinput_lib

    def get(self) -> Optional[Tuple]:
        data = MouseScrollData()
        if self.rawinput_lib.cdll.get_mouse_scroll_input(ctypes.byref(data)):
            return data.timestamp, data.scrollAmount

class KeyboardInputReader:
    def __init__(self, rawinput_lib: RawInputLib):
        self.rawinput_lib = rawinput_lib

    def get(self) -> Optional[Tuple]:
        data = KeyboardData()
        if self.rawinput_lib.cdll.get_keyboard_input(ctypes.byref(data)):
            return data.timestamp, data.keyCode, data.down

@dataclass
class Empty:
    pass

class PeekableReader:
    def __init__(self, reader):
        self.reader = reader
        self.peeked_data: Union[Optional[Tuple], Empty] = Empty()

    def get(self) -> Optional[Tuple]:
        if isinstance(self.peeked_data, Empty):
            return self.reader.get()
        data = self.peeked_data
        self.peeked_data = Empty()
        return data

    def peek(self) -> Optional[Tuple]:
        if isinstance(self.peeked_data, Empty):
            self.peeked_data = self.reader.get()
        return self.peeked_data

class RawInputReader:
    def __init__(self, rawinput_lib = RawInputLib()):
        self.rawinput_lib = rawinput_lib
        self.mouse_move = PeekableReader(MouseMoveInputReader(rawinput_lib))
        self.mouse_button = PeekableReader(MouseButtonInputReader(rawinput_lib))
        self.mouse_scroll = PeekableReader(MouseScrollInputReader(rawinput_lib))
        self.keyboard = PeekableReader(KeyboardInputReader(rawinput_lib))

    @property
    def inputs(self):
        return [self.mouse_move, self.mouse_button, self.mouse_scroll, self.keyboard]

    def open(self) -> bool:
        """
        Get ready to read raw input, starts C process
        Returns bool indicating success
        """
        return self.rawinput_lib.cdll.initialize_raw_input()

    def close(self):
        """
        Close reading process and release handlers
        """
        self.rawinput_lib.cdll.cleanup_raw_input()

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
            for input in reader.inputs:
                if input.peek() != None:
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
            for input in reader.inputs:
                if input.peek() != None:
                    idle_counter = 0
                    break

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
            kb_info = self.reader.keyboard.peek()
            if kb_info is None: # Nothing pressed
                continue
            keycode, keydown = kb_info[1], kb_info[2]
            if not keydown: # Not a down event
                continue
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
