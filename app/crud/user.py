from typing import Optional

from app.crud.base import CrudBase
from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class CrudUser(CrudBase[User, UserCreate, UserUpdate]):
    async def get_by_email(self, session: AsyncSession, email: str) -> Optional[User]:
        return (await session.execute(select(self.model).where(self.model.email == email))).scalars().first()

    async def get_by_username(self, session: AsyncSession, username: str) -> Optional[User]:
        return (await session.execute(select(self.model).where(self.model.username == username))).scalars().first()

    async def create_user(self, session: AsyncSession, *, obj_in: UserCreate) -> User:
        db_obj = User(
            username=obj_in.username,
            password=get_password_hash(obj_in.password),
            email=obj_in.email,
            permission_level=obj_in.permission_level
        )

        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)

        return db_obj

    async def authenticate(self, session: AsyncSession, *, username: str, password: str) -> Optional[User]:
        db_user = await self.get_by_username(session, username)

        if not db_user:
            return False

        if not verify_password(password, db_user.password):
            return False

        return db_user


user = CrudUser(User)
