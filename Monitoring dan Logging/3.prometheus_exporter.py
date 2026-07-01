from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import time
import random
import threading


prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Latency of model predictions in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests'
)

prediction_errors_total = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

prediction_value = Gauge(
    'prediction_value',
    'Current predicted goals value'
)

prediction_confidence = Gauge(
    'model_confidence',
    'Current model confidence score'
)

goals_mean = Gauge(
    'goals_mean',
    'Mean of predicted goals over window'
)

goals_std = Gauge(
    'goals_std',
    'Std of predicted goals over window'
)

model_memory_usage = Gauge(
    'model_memory_usage_bytes',
    'Estimated memory usage of the model'
)

request_size_bytes = Summary(
    'request_size_bytes',
    'Size of prediction request payload'
)

prediction_by_position = Gauge(
    'prediction_by_position',
    'Prediction count per position',
    ['position']
)


def generate_metrics():
    while True:
        prediction_requests_total.inc(random.randint(0, 5))
        prediction_value.set(random.uniform(0, 3))
        prediction_confidence.set(random.uniform(0.85, 0.99))
        goals_mean.set(random.uniform(0.1, 1.5))
        goals_std.set(random.uniform(0.3, 0.8))
        model_memory_usage.set(random.uniform(100, 500) * 1024 * 1024)
        prediction_by_position.labels(position='Defender').set(random.randint(100, 500))
        prediction_by_position.labels(position='Forward').set(random.randint(100, 500))
        prediction_by_position.labels(position='Midfielder').set(random.randint(100, 500))
        prediction_by_position.labels(position='Goalkeeper').set(random.randint(100, 500))

        latency = random.uniform(0.05, 0.8)
        prediction_latency.observe(latency)
        request_size_bytes.observe(random.randint(500, 5000))

        if random.random() < 0.05:
            prediction_errors_total.inc()

        time.sleep(5)


if __name__ == '__main__':
    print('Starting Prometheus exporter on port 8001...')
    threading.Thread(target=generate_metrics, daemon=True).start()
    start_http_server(8001)
    print('Exporter running at http://localhost:8001/metrics')
    while True:
        time.sleep(1)
