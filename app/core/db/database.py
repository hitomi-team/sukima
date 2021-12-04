import databases
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


database = databases.Database(settings.get_url())

engine = create_engine(
    settings.get_url(), connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

