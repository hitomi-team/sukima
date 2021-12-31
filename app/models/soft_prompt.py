from app.core.config import settings
from app.db.base_class import Base
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, Numeric


class SoftPrompt(Base):
    __tablename__ = "soft_prompts"

    id = Column(String, primary_key=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    public = Column(Boolean, nullable=False, default=True)
    creator = Column(Integer, ForeignKey("users.id"), nullable=False)
    model = Column(String, nullable=False)
    # model = Column(Integer, ForeignKey("models.id"), nullable=False)
    loss = Column(Numeric, nullable=False)
    steps = Column(Integer, nullable=False)

    def storage_path(self) -> str:
        return settings.STORAGE_PATH / f"{self.id}.zz"

    def read(self) -> bytes:
        with open(self.storage_path(), "rb") as data_file:
            return data_file.read()

    def write(self, tensor_data: bytes):
        with open(self.storage_path(), "wb") as data_file:
            data_file.write(tensor_data)

    def asdict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "public": self.public,
            "model": self.model,
            "loss": self.loss,
            "steps": self.steps,
        }
