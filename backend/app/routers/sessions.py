from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import AttendanceSession
from app.schemas import SessionCreate, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

@router.post("", response_model=SessionOut, status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    if payload.ends_at <= payload.starts_at:
        raise HTTPException(status_code=400, detail="Session end must be after start")
    session = AttendanceSession(**payload.model_dump())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db)):
    return db.query(AttendanceSession).order_by(AttendanceSession.starts_at.desc()).all()
