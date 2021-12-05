from sqlalchemy import Boolean, Column, ForeignKey, Integer, MetaData, String, Table

metadata = MetaData()

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


models = Table(
    'models',
    metadata,
    Column('model_name', String, primary_key=True, index=True),
    Column('size', Integer)
)
