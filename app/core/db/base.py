# Import models first so that Base will have them before being imported by Alembic
from app.core.db.database import Base  # noqa
from app.core.db.models import User, ModelItem  # noqa