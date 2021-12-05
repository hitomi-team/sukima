import databases
from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine

database = databases.Database(settings.DATABASE_URI)

engine = create_async_engine(
    settings.DATABASE_URI, pool_pre_ping=True
)
