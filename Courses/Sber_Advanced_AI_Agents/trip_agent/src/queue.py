from dataclasses import dataclass

from src import providers


@dataclass
class BookingTask:
    message_id: str
    idempotency_key: str
    destination: str
    dates: str
    flight: str
    hotel: str
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3


class BookingQueue:
    """In-process publish/reserve/save/ack pipeline for booking, with priority, bounded
    retry, a DLQ, and separate dedupe by message_id (delivery) vs idempotency_key (effect)."""

    def __init__(self):
        self._pending: list[BookingTask] = []
        self._seen_message_ids: set[str] = set()
        self._results: dict[str, str] = {}
        self._acked: set[str] = set()
        self.dlq: list[BookingTask] = []

    def _enqueue(self, task: BookingTask) -> None:
        self._pending.append(task)
        self._pending.sort(key=lambda t: t.priority)

    def publish(self, task: BookingTask) -> bool:
        if task.message_id in self._seen_message_ids:
            return False
        self._seen_message_ids.add(task.message_id)
        self._enqueue(task)
        return True

    async def _reserve(self, task: BookingTask) -> str:
        if task.idempotency_key in self._results:
            return self._results[task.idempotency_key]
        return await providers.booking_provider.reserve(task.destination, task.dates, task.flight, task.hotel)

    def _save_result(self, task: BookingTask, confirmation: str) -> None:
        self._results[task.idempotency_key] = confirmation

    def _ack(self, task: BookingTask) -> None:
        self._acked.add(task.idempotency_key)

    async def run_worker_once(self) -> None:
        if not self._pending:
            return
        task = self._pending.pop(0)

        try:
            confirmation = await self._reserve(task)
        except Exception:
            task.attempts += 1
            if task.attempts < task.max_attempts:
                self._enqueue(task)
            else:
                self.dlq.append(task)
            return

        self._save_result(task, confirmation)
        self._ack(task)

    def result_for(self, idempotency_key: str) -> str | None:
        return self._results.get(idempotency_key)

    def is_acked(self, idempotency_key: str) -> bool:
        return idempotency_key in self._acked

    def dlq_reason(self, idempotency_key: str) -> str | None:
        for task in self.dlq:
            if task.idempotency_key == idempotency_key:
                return f"gave up after {task.attempts} attempts"
        return None


default_queue = BookingQueue()
