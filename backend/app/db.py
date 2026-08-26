from collections.abc import Mapping
from typing import Any

import certifi
from gridfs import AsyncGridFSBucket
from pymongo import ASCENDING, AsyncMongoClient, IndexModel
from pymongo.server_api import ServerApi
from bson import ObjectId

from .settings import Settings


class MongoStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: AsyncMongoClient | None = None
        self.database = None
        self.bucket: AsyncGridFSBucket | None = None

    @property
    def configured(self) -> bool:
        return self.client is not None and self.database is not None and self.bucket is not None

    async def startup(self) -> None:
        if not self.settings.database_configured:
            return

        self.client = AsyncMongoClient(
            self.settings.mongodb_uri,
            server_api=ServerApi("1"),
            tlsCAFile=certifi.where(),
            tz_aware=True,
        )
        await self.client.admin.command({"ping": 1})
        self.database = self.client[self.settings.mongodb_database]
        self.bucket = AsyncGridFSBucket(self.database, bucket_name=self.settings.mongodb_gridfs_bucket)
        await self._ensure_indexes()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
            self.database = None
            self.bucket = None

    async def _ensure_indexes(self) -> None:
        if self.database is None:
            return

        await self.database.chats.create_indexes([
            IndexModel([("updatedAt", ASCENDING)], name="chats_updatedAt"),
        ])
        await self.database.analysis_runs.create_indexes([
            IndexModel(
                [("chatId", ASCENDING), ("createdAt", ASCENDING)],
                name="analysis_runs_chatId_createdAt",
            ),
        ])
        await self.database.files.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("normalizedName", ASCENDING)],
                name="files_runId_normalizedName",
                unique=True,
            ),
            IndexModel(
                [("chatId", ASCENDING), ("createdAt", ASCENDING)],
                name="files_chatId_createdAt",
            ),
        ])
        await self.database.chart_revisions.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("createdAt", ASCENDING)],
                name="chart_revisions_runId_createdAt",
            ),
        ])
        await self.database.messages.create_indexes([
            IndexModel(
                [("chatId", ASCENDING), ("runId", ASCENDING), ("createdAt", ASCENDING)],
                name="messages_chatId_runId_createdAt",
            ),
        ])
        chart_row_index_cursor = await self.database.chart_rows.list_indexes()
        existing_chart_row_indexes = await chart_row_index_cursor.to_list(length=None)
        for index in existing_chart_row_indexes:
            if (
                index.get("name") == "chart_rows_revisionId_rowIndex"
                and not index.get("unique", False)
            ):
                await self.database.chart_rows.drop_index(index["name"])
        await self.database.chart_rows.create_indexes([
            IndexModel(
                [("revisionId", ASCENDING), ("rowIndex", ASCENDING)],
                name="chart_rows_revisionId_rowIndex",
                unique=True,
            ),
        ])
        await self.database.supporting_documents.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("fileId", ASCENDING)],
                name="supporting_documents_runId_fileId",
            ),
        ])
        await self.database.verification_results.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("rowId", ASCENDING)],
                name="verification_results_runId_rowId",
            ),
        ])
        await self.database.audit_events.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("createdAt", ASCENDING)],
                name="audit_events_runId_createdAt",
            ),
        ])
        await self.database.langgraph_checkpoints.create_indexes([
            IndexModel(
                [("runId", ASCENDING), ("createdAt", ASCENDING)],
                name="langgraph_checkpoints_runId_createdAt",
            ),
        ])
        await self.database.agent_actions.create_indexes([
            IndexModel(
                [("chatId", ASCENDING), ("runId", ASCENDING), ("createdAt", ASCENDING)],
                name="agent_actions_chatId_runId_createdAt",
            ),
            IndexModel(
                [("chatId", ASCENDING), ("status", ASCENDING)],
                name="agent_actions_chatId_status",
            ),
        ])

    async def insert(self, collection: str, document: Mapping[str, Any]) -> ObjectId:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        result = await self.database[collection].insert_one(dict(document))
        return result.inserted_id

    async def insert_many(self, collection: str, documents: list[Mapping[str, Any]]) -> list[ObjectId]:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        if not documents:
            return []
        result = await self.database[collection].insert_many([dict(document) for document in documents])
        return list(result.inserted_ids)

    async def update(self, collection: str, filter_query: Mapping[str, Any], update: Mapping[str, Any]) -> None:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        await self.database[collection].update_one(dict(filter_query), dict(update))

    async def compare_and_set(
        self,
        collection: str,
        filter_query: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> bool:
        """Apply an update only when the current document still matches the filter."""
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        result = await self.database[collection].update_one(dict(filter_query), dict(update))
        return result.modified_count == 1

    async def update_many(
        self,
        collection: str,
        filter_query: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> None:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        await self.database[collection].update_many(dict(filter_query), dict(update))

    async def find_many(self, collection: str, filter_query: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        return [document async for document in self.database[collection].find(dict(filter_query))]

    async def find_one(self, collection: str, filter_query: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        return await self.database[collection].find_one(dict(filter_query))

    async def find_recent_chats(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        cursor = self.database.chats.find({}).sort("updatedAt", -1).limit(limit)
        return [document async for document in cursor]

    async def find_latest_run(self, chat_id: ObjectId) -> dict[str, Any] | None:
        if self.database is None:
            raise RuntimeError("MongoDB is not configured")
        cursor = self.database.analysis_runs.find({"chatId": chat_id}).sort("createdAt", -1).limit(1)
        runs = await cursor.to_list(length=1)
        return runs[0] if runs else None

    async def read_file(self, gridfs_id: ObjectId) -> bytes:
        if self.bucket is None:
            raise RuntimeError("MongoDB is not configured")
        stream = await self.bucket.open_download_stream(gridfs_id)
        return await stream.read()

    async def delete_many_by_ids(self, collection: str, ids: list[ObjectId]) -> None:
        if self.database is not None and ids:
            await self.database[collection].delete_many({"_id": {"$in": ids}})

    async def store_file(
        self,
        *,
        filename: str,
        content_type: str | None,
        metadata: Mapping[str, Any],
        upload_file: Any,
        max_bytes: int,
    ) -> tuple[ObjectId, int]:
        if self.database is None or self.bucket is None:
            raise RuntimeError("MongoDB is not configured")

        gridfs_id = ObjectId()
        grid_in = self.bucket.open_upload_stream_with_id(
            gridfs_id,
            filename,
            metadata={**dict(metadata), "contentType": content_type},
        )
        total_bytes = 0
        try:
            while chunk := await upload_file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes >= max_bytes:
                    raise ValueError(f"{filename} exceeds the configured file-size limit")
                await grid_in.write(chunk)
            await grid_in.close()
            return gridfs_id, total_bytes
        except Exception:
            await grid_in.abort()
            raise

    async def store_bytes(
        self,
        *,
        filename: str,
        content_type: str,
        metadata: Mapping[str, Any],
        data: bytes,
    ) -> tuple[ObjectId, int]:
        if self.database is None or self.bucket is None:
            raise RuntimeError("MongoDB is not configured")

        gridfs_id = ObjectId()
        grid_in = self.bucket.open_upload_stream_with_id(
            gridfs_id,
            filename,
            metadata={**dict(metadata), "contentType": content_type},
        )
        try:
            await grid_in.write(data)
            await grid_in.close()
            return gridfs_id, len(data)
        except Exception:
            await grid_in.abort()
            raise
