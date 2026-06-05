"""
PostgreSQL implementation of CheckpointSaver for enterprise persistence.
"""

from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
from asera.memory.checkpointer import Checkpoint, BaseCheckpointSaver
from asera.logging import logger


class PostgresCheckpointSaver(BaseCheckpointSaver):
    """
    PostgreSQL persistence for conversation memory using asyncpg.
    Persists messages with thread isolation and incremental saves.
    """

    def __init__(self, pool_getter_or_pool: Any):
        """
        Args:
            pool_getter_or_pool: An asyncpg Pool or an async callable returning a Pool.
        """
        self._pool_source = pool_getter_or_pool

    async def _get_pool(self):
        if callable(self._pool_source):
            return await self._pool_source()
        return self._pool_source

    async def save(self, checkpoint: Checkpoint) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                'SELECT count(*) FROM "asera_messages" WHERE "thread_id" = $1',
                checkpoint.thread_id
            ) or 0

            messages_to_save = checkpoint.messages[count:]
            if not messages_to_save:
                return

            now = datetime.utcnow()
            vals = []
            for msg in messages_to_save:
                vals.append((
                    str(uuid.uuid4()),
                    checkpoint.thread_id,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    json.dumps(msg.get("metadata", {})),
                    now
                ))

            await conn.executemany(
                '''
                INSERT INTO "asera_messages" ("id", "thread_id", "role", "content", "metadata", "created_at")
                VALUES ($1, $2, $3, $4, $5, $6)
                ''',
                vals
            )
            logger.debug(f"Saved {len(messages_to_save)} messages to Postgres for thread '{checkpoint.thread_id}'")

    async def load(self, thread_id: str) -> Optional[List[Dict[str, Any]]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                '''
                SELECT "role", "content", "metadata"
                FROM "asera_messages"
                WHERE "thread_id" = $1
                ORDER BY "created_at" ASC
                ''',
                thread_id
            )
            if not rows:
                return None

            messages = []
            for r in rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                messages.append({
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": meta
                })
            return messages

    async def clear(self, thread_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM "asera_messages" WHERE "thread_id" = $1',
                thread_id
            )
            logger.debug(f"Deleted Postgres messages for thread '{thread_id}'")
