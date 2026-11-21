from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field

class UserCreate(BaseModel):
    student_id: str = Field(min_length=1, max_length=50)
    full_name: str = Field(min_length=2, max_length=150)
    email: str = Field(min_length=5, max_length=255)
    department: str = ""
    section: str = ""

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    student_id: str
    full_name: str
    email: str
    department: str
    section: str

class SessionCreate(BaseModel):
    title: str
    starts_at: datetime
    ends_at: datetime

class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    starts_at: datetime
    ends_at: datetime
    status: str
