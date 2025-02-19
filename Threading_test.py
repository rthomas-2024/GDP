import time
import numpy as np

dt = 0.1  # Interval in seconds
start_time = time.perf_counter()
n = 0  # Step counter

while True:
    elapsed_time = start_time - time.perf_counter()
    current_time = time.perf_counter()

    # Generate random matrices (3x3)
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)

    # Perform matrix multiplication
    C = np.dot(A, B)

    # Print the result along with the precise timestamp
    print(f"Time: {elapsed_time:.6f}, Matrix Result:\n{C}\n")

    # Compute the exact next scheduled time
    n += 1
    next_time = start_time + n * dt

    # Busy-wait until the next precise moment
    while time.perf_counter() < next_time: # note this code is high CPU usage. the "pass" argument keeps the loop running until the the current time is exceeds next_time
        pass  # Active waiting for precision
