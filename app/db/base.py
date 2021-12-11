# Import models first so that Base will have them before being imported by Alembic
from app.db.base_class import Base  # noqa
from app.v1.models import User, ModelItem  # noqa