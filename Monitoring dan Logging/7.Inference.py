import requests
import json
import random
import time


BASE_URL = 'http://localhost:8000'


def test_health():
    response = requests.get(f'{BASE_URL}/health')
    print(f'Health: {response.json()}')
    return response.ok


def test_predict(features=None):
    if features is None:
        features = [random.uniform(0, 1) for _ in range(59)]

    payload = {'features': features}
    response = requests.post(f'{BASE_URL}/predict', json=payload)
    result = response.json()
    print(f'Prediction: {json.dumps(result, indent=2)}')
    return result


def test_metrics():
    response = requests.get(f'{BASE_URL}/metrics')
    print(f'Metrics (first 500 chars): {response.text[:500]}')
    return response.ok


if __name__ == '__main__':
    print('=== Testing Model Serving ===\n')

    print('1. Health Check:')
    test_health()
    print()

    print('2. Metrics Endpoint:')
    test_metrics()
    print()

    print('3. Multiple Predictions (with monitoring load):')
    for i in range(10):
        print(f'\nPrediction {i+1}:')
        result = test_predict()
        time.sleep(0.5)

    print('\n=== Done ===')
