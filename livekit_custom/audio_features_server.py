import os
import asyncio
import multiprocessing

# Server code
def server_process(pipe_name):
    with open(pipe_name, 'r') as pipe:
        while True:
            data = pipe.readline()
            if not data:
                break
            # Process the data here
            print(f"Received data: {data}")

def start_server(pipe_name, num_processes):
    os.mkfifo(f"./pipes/{pipe_name}")
    processes = []
    for _ in range(num_processes):
        process = multiprocessing.Process(target=server_process, args=(pipe_name,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()