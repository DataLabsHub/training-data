# Bundled component notice

This challenge baseline is a self-contained integration of the following
runtime components.  The original license files remain in each component.

| Component | Bundled snapshot | License | Role |
|---|---|---|---|
| Celery | challenge baseline | BSD-3-Clause | application, task, canvas, worker, result, event, and scheduler runtime |
| Kombu | challenge baseline | BSD-3-Clause | message, producer/consumer, QoS, connection, and transport runtime |
| Billiard | challenge baseline | BSD-3-Clause | worker process and pool lifecycle runtime |
| Vine | challenge baseline | BSD-3-Clause | promise/callback runtime used by Celery and Kombu |

The components are maintained as one challenge workspace and are imported
directly from this repository.  No network checkout is performed at build,
solve, or verification time.
