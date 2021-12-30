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

    def storage_filename(self) -> str:
        return f"{self.id}.zz"

    def asdict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "public": self.public,
            "model": self.model,
            "loss": self.loss,
            "step": self.steps,
            "url": f"/storage/{self.storage_filename()}"
        }

