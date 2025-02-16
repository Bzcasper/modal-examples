import pytest
import requests
import json
import time

# Define URL and audio files
URL = "https://modal-labs--faster-whisper-model-web.modal.run"

AUDIO_FILES = {
    "thirty": "wavs/thirty_compressed.wav",
    # "short": "wavs/short.wav",
    # "preamble": "wavs/preamble.wav",
    "long": "wavs/long.wav"
}

SCENARIOS = ["thirty"]
        
# Fixture to perform HTTP POST request
@pytest.fixture
def perform_request(benchmark):
    def _perform_request(audio_type, bench=True):

        # Warm up the TCP connection
        session = requests.Session()
        # for _ in range(5):
        #     print(f"Starting request at {time.monotonic()}")
        #     session.post(URL, files = {"file": (AUDIO_FILES["long"], open(AUDIO_FILES["long"], "rb"), "audio/wav")})
        
        def fn():
            files = {"file": (AUDIO_FILES[audio_type], open(AUDIO_FILES[audio_type], "rb"), "audio/wav")}

            return session.post(URL, files=files)
        if bench:
            return benchmark(fn)
        else:
            return fn()

    return _perform_request

# Benchmark test function
@pytest.mark.parametrize("audio_type", SCENARIOS)
def test_bench(perform_request, audio_type):
    result = perform_request(audio_type)
    assert result.status_code == 200  # Assert successful response