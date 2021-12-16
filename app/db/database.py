from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    settings.DATABASE_URI, pool_pre_ping=True
)

async_session = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)
