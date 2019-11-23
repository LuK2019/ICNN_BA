from multiprocessing import Process
import time


def thread_function():
    print("Thread_function started.")
    time.sleep(3)
    print("Thread_function executed.")
    return 3


if __name__ == "__main__":
    # We create a Process
    action_process = Process(target=thread_function)

    # We start the process and we block for 5 seconds.
    action_process.start()
    action_process.join(timeout=5)

    # We terminate the process.
    action_process.terminate()
    print("ACTION", jops.eaction_process)
    print("Hey there! I timed out! You can do things after me!")
