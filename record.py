import asyncio
from vg_control.main import Recorder

async def main():
    recorder = Recorder()
    try:
        await recorder.main()
    except KeyboardInterrupt:
        print("Recording stopped by user")

if __name__ == "__main__":
    asyncio.run(main())