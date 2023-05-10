import threading

count = 0


def run(lock):
    for i in range(5):
        with lock:
            global count
            count = count + 1


lock = threading.Lock()
thread1 = threading.Thread(target=run, args=(lock,))
thread2 = threading.Thread(target=run, args=(lock,))

thread1.start()
thread2.start()

thread1.join()
thread2.join()
print(count)
