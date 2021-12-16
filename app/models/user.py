from app.db.base_class import Base
from sqlalchemy import Column, ForeignKey, Integer, SmallInteger, String, Table

user_model_association = Table(
    'user_model_association',
    Base.metadata,
    Column('user_id', ForeignKey('users.id')),
    Column('model_id', ForeignKey('models.id'))
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    permission_level = Column(SmallInteger, default=0, nullable=False)


class Model(Base):
    __tablaname__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    size = Column(Integer, nullable=False)
