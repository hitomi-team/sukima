from sqlalchemy import Table, Column, String, Integer

from app.models.user import metadata

models = Table(
    'models',
    metadata,
    Column('model_name', String, primary_key=True, index=True),
    Column('size', Integer)
)