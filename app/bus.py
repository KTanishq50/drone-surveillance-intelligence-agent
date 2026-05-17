import asyncio

observation_queue: asyncio.Queue = None
result_queue: asyncio.Queue = None

def init():
    global observation_queue, result_queue
    observation_queue = asyncio.Queue()
    result_queue      = asyncio.Queue()