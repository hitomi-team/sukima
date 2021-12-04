import databases
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


database = databases.Database(settings.SQLALCHEMY_DATABASE_URI)

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
