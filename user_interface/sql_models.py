from flask_login import UserMixin, AnonymousUserMixin
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask import session
from uuid import uuid4
import json

db = SQLAlchemy()

#custom anonymous user with a random id to avoid errors
class AnonymousUser(AnonymousUserMixin):
    def __init__(self):
        super().__init__()
        if "anonymus_id" not in session:
            session["anonymus_id"] = str(uuid4())
        self.id = session["anonymus_id"]
        self.permission = "guest"

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    permission = db.Column(db.String(20), nullable=False, default="user")
    conversations = db.relationship('Conversation', backref='author', lazy=True)
    
    def __repr__(self) -> str:
        return f"User('{self.username}', '{self.email}', '{self.permission}')"

class UserConfig(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    key = db.Column(db.String(20), nullable=False)
    value = db.Column(db.String(120), nullable=False)
    # set user_id and key as primary key
    __table_args__ = (
        db.PrimaryKeyConstraint('user_id', 'key'),
    )

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class ConversationMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class ConversationConfig(db.Model):
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    key = db.Column(db.String(20), nullable=False)
    value = db.Column(db.JSON, nullable=False)
    # set conversation_id and key as primary key
    __table_args__ = (
        db.PrimaryKeyConstraint('conversation_id', 'key'),
    )
    
    def __repr__(self) -> str:
        return f"ConversationConfig('{self.conversation_id}', '{self.key}', '{self.value}')"

class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self) -> str:
        return f"ContactMessage('{self.name}', '{self.email}', '{self.content}')"

class SystemConfig(db.Model):
    key = db.Column(db.String(20), primary_key=True)
    value = db.Column(db.JSON, nullable=False)
    
    def __repr__(self) -> str:
        return f"SystemConfig('{self.key}', '{self.value}')"