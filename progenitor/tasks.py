from celery import shared_task


@shared_task(name="progenitor.ping")
def ping() -> str:
    """Round-trip check for the broker, worker, and result backend."""
    return "pong"
