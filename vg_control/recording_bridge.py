import asyncio
import json
import sys
from .recorder import SimpleRecorder, main as recorder_main
from .input_tracking.rawinputlib import HotkeyManager, RAW_INPUT
from .input_tracking.keybinds import CODE_TO_KEY

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
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Recording Bridge')
    parser.add_argument('--start-key', type=str, default='F4', help='Start recording hotkey')
    parser.add_argument('--stop-key', type=str, default='F5', help='Stop recording hotkey')
    
    # Parse args (skip the first arg which is the script name)
    args = parser.parse_args()
    
    start_key = args.start_key
    stop_key = args.stop_key
    
    # Print detected keys
    print(f"Recording bridge using hotkeys: Start={start_key}, Stop={stop_key}")
    sys.stdout.flush()
    
    bridge = RecorderBridge(start_key, stop_key)
    await bridge.run()

if __name__ == "__main__":
    asyncio.run(bridge_main())