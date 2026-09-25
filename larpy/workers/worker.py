"""RQ Worker for price estimation tasks."""

import sys
import os
from redis import Redis
from rq import Worker, Queue, Connection

from larpy.config.settings import app_config
from larpy.workers.tasks import price_queue


def listen_on_queues():
    """Start listening on RQ queues."""
    return [price_queue]


class Worker:
    """Custom RQ Worker with graceful shutdown."""

    def __init__(self, queues=None):
        self.queues = queues or listen_on_queues()
        self.redis_conn = Redis(
            host=app_config.redis.host,
            port=app_config.redis.port,
            db=app_config.redis.db,
            decode_responses=True,
        )

    def start(self):
        """Start the worker."""
        print(f"Starting Larpy Worker on queues: {[q.name for q in self.queues]}")
        print(f"   Redis: {app_config.redis.host}:{app_config.redis.port}")
        print(f"   Model: {app_config.model.name}")
        print(f"   Press Ctrl+C to stop\n")

        with Connection(self.redis_conn):
            worker = Worker(
                self.queues,
                name="larpy-price-worker",
            )
            worker.work()

    def stop(self):
        """Stop the worker gracefully."""
        print("\nShutting down worker...")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Larpy RQ Worker")
    parser.add_argument("--queues", nargs="+", default=["price_estimation"], help="Queue names")
    parser.add_argument("--burst", action="store_true", help="Run in burst mode")

    args = parser.parse_args()

    queue_list = [Queue(q, connection=Redis(host=app_config.redis.host, port=app_config.redis.port)) for q in args.queues]
    worker = Worker(queue_list)
    worker.start()


if __name__ == "__main__":
    main()
