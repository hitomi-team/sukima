from sqlalchemy import Table, Column, ForeignKey, String, Boolean

from app.db.base_class import Base

metadata = Base.metadata


user_model_association = Table(
    'user_model_association',
    metadata,
    Column('username', ForeignKey('users.username')),
    Column('model_name', ForeignKey('models.model_name'))
)


users = Table(
    'users',
    metadata,
    Column('username', String, primary_key=True, index=True),
    Column('password', String),
    Column('approved', Boolean, default=False)
)