import asyncio
import json
import sys
from vg_control.recorder import SimpleRecorder, main as recorder_main
from vg_control.input_tracking.rawinputlib import HotkeyManager, RAW_INPUT
from vg_control.input_tracking.keybinds import CODE_TO_KEY

class RecorderBridge:
    def __init__(self, start_key, stop_key):
        self.recorder = SimpleRecorder()
        self.hotkeys = HotkeyManager()
        
        # Register the custom hotkeys
        self.hotkeys.add_callback(start_key, self.recorder.start_recording)
        self.hotkeys.add_callback(stop_key, self.recorder.stop_recording)
    
    async def run(self):
        try:
            # Print ready status for Node
            print(json.dumps({"status": "ready"}))
            sys.stdout.flush()
            
            # Run the hotkey event loop
            await self.hotkeys.event_loop()
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}))
            sys.stdout.flush()
        finally:
            RAW_INPUT.close()

async def bridge_main():
    # Get hotkeys from command line args
    if len(sys.argv) != 3:
        start_key = 'F4'
        stop_key = 'F5'
    else:
        # Check if provided keys are in CODE_TO_KEY mapping
        try:
            start_code = int(sys.argv[1])
            stop_code = int(sys.argv[2])
            start_key = CODE_TO_KEY.get(start_code, 'F4')
            stop_key = CODE_TO_KEY.get(stop_code, 'F5')
        except ValueError:
            # If not valid integers, use defaults
            start_key = 'F4'
            stop_key = 'F5'
    bridge = RecorderBridge(start_key, stop_key)
    await bridge.run()

if __name__ == "__main__":
    asyncio.run(bridge_main())