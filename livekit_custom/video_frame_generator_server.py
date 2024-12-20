import os
import asyncio
import multiprocessing

# Server code
def server_process(pipe_name):
    try:
        with open(pipe_name, 'r') as pipe:
                while True:
                    data = pipe.readline()
                    if not data:
                        break
                    # Process the data here
                    print(f"Received data: {data}")
    except Exception as e:
        print(f"Pipe : An error occured : {e}")
    print(f"Pipe gone {pipe_name}")

async def start_server(type, num_processes):
    processes = []
    try:
        for i in range(num_processes):
            pipe_name = f"./pipes/{type}/{i}"
            if not os.path.exists(pipe_name):
                os.makedirs(pipe_name)
            pipe = os.path.join(pipe_name,type)
            if not os.path.exists(pipe):
                os.mkfifo(pipe)
            process = multiprocessing.Process(target=server_process, args=(pipe,), name=pipe)
            process.start()
            processes.append(process)
            # print(f"Process {pipe} started")
        for process in processes:
            try:
                print(f"Process {process.name} starting")
                asyncio.get_event_loop().run_in_executor(None,process.join)  # Wait for the process to finish
                print(f"Process {process.name} started")
            except Exception as e:
                print(f"Processor : Keyboard Interruption")
    except Exception as e:
        
        for i in range(len(processes)):
            print(f"Processor : An error occured : {e}")
            pid = processes.pop()
            if pid is None:
                break
            name = pid.name
            pid.terminate()
            pid.join()
            print(f"PID {name} closed")

            pipe_name = f"./pipes/{type}/{i}"
            pipe = os.path.join(pipe_name,type)
            if os.path.exists(pipe):
                os.remove(pipe)
                # os.removedirs(pipe_name)

if __name__ == "__main__":
    num_processes = 4
    type = "video_frame_generator"
    print(f"Server ({type}) starting")
    try:
        asyncio.run(start_server(type,num_processes))
    except Exception as e:
        print(f"Exception catched : {e}")
    print(f"Server ({type}) stopped")
    
    