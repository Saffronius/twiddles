from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, CheckConstraint, Integer, Float, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import uuid

from ..database import Base

class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"), nullable=True)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    description_json = Column(JSONB, default=dict)
    system_prompt = Column(Text, default="")
    inherit_prompt = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    embedding_centroid = Column(Vector(1536), nullable=True)
    
    # Relationships
    parent = relationship("Channel", remote_side=[id], back_populates="children")
    children = relationship("Channel", back_populates="parent")
    messages = relationship("Message", back_populates="channel")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"), nullable=False)
    role = Column(Text, CheckConstraint("role IN ('user', 'assistant', 'system')"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    embedding = Column(Vector(1536), nullable=True)
    memory_layer = Column(Text, CheckConstraint("memory_layer IN ('recent', 'compressed')"), default='recent')
    scratchpad_origin_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Relationships
    channel = relationship("Channel", back_populates="messages")

class ScratchpadEntry(Base):
    __tablename__ = "scratchpad_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_json = Column(JSONB, nullable=False)
    content_text = Column(Text, nullable=False)
    state = Column(Text, CheckConstraint("state IN ('staged', 'routed')"), default='staged')
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class RoutingLog(Base):
    __tablename__ = "routing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entry_id = Column(UUID(as_uuid=True), ForeignKey("scratchpad_entries.id"), nullable=False)
    target_channel_id = Column(UUID(as_uuid=True), nullable=True)
    confidence = Column(Float, nullable=True)
    routed_by = Column(Text, nullable=False)  # 'user' or 'router'
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class MemorySummary(Base):
    __tablename__ = "memory_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"), nullable=False)
    source_message_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class ContextSlice(Base):
    __tablename__ = "context_slices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"), nullable=False)
    user_message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    included_message_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    included_summary_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    mode = Column(Text, CheckConstraint("mode IN ('plain', 'rag')"), nullable=False)
    token_estimate = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class KnowledgeCard(Base):
    __tablename__ = "knowledge_cards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"), nullable=False)
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    source_message_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow) 