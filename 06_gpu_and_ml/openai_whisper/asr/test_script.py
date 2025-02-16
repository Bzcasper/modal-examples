import requests
import time
from random import randbytes
import numpy as np

short = randbytes(1 * 1024 * 1024)
long = randbytes(12 * 1024 * 1024)
URL = "https://modal-labs--latency-test-crazy-final-web.modal.run"
session = requests.Session()

# warm up TCP connection
for _ in range(5):
    session.post(URL, data=long)

differences = []
for _ in range(10):
    start = time.monotonic() 
    session.post(URL, data=short)
    end = time.monotonic()
    differences.append(end - start)

print(np.percentile(differences, 50), np.percentile(differences, 90))

