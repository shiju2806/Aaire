"""
Authentication and Authorization for AAIRE - MVP-FR-021 through MVP-FR-024
SAML 2.0 SSO integration with role-based access control
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import structlog

from fastapi import HTTPException, status
from passlib.context import CryptContext

logger = structlog.get_logger()

@dataclass
class User:
    id: str
    email: str
    name: str
    roles: list
    is_admin: bool
    can_upload_documents: bool
    context: Dict[str, Any]
    session_expires: datetime

class AuthManager:
    def __init__(self):
        """Initialize authentication manager"""
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT settings
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_hours = 8  # As specified in MVP
        
        # In-memory user store for MVP (replace with database in production)
        self.users_db = {
            "admin@company.com": {
                "id": "admin-001",
                "email": "admin@company.com",
                "name": "Admin User",
                "hashed_password": self.pwd_context.hash("admin123"),  # Demo only
                "roles": ["admin", "user"],
                "is_admin": True,
                "can_upload_documents": True
            },
            "user@company.com": {
                "id": "user-001", 
                "email": "user@company.com",
                "name": "Regular User",
                "hashed_password": self.pwd_context.hash("user123"),  # Demo only
                "roles": ["user"],
                "is_admin": False,
                "can_upload_documents": False
            }
        }
        
        # Active sessions
        self.active_sessions = {}
        
        logger.info("Authentication manager initialized", 
                   user_count=len(self.users_db))
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        
        expire = datetime.utcnow() + timedelta(hours=self.access_token_expire_hours)
        to_encode = {
            "sub": user_data["email"],
            "user_id": user_data["id"],
            "name": user_data["name"],
            "roles": user_data["roles"],
            "exp": expire
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email/password"""
        
        user = self.users_db.get(email)
        if not user:
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            return None
        
        return user
    
    async def authenticate(self, token: str) -> User:
        """
        Authenticate user from token - MVP-FR-021, MVP-FR-023
        """
        
        # Verify token
        payload = self.verify_token(token)
        
        # Get user data
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user_data = self.users_db.get(email)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create user object
        user = User(
            id=user_data["id"],
            email=user_data["email"],
            name=user_data["name"],
            roles=user_data["roles"],
            is_admin=user_data["is_admin"],
            can_upload_documents=user_data["can_upload_documents"],
            context={
                "department": "unknown",  # Would come from SAML in production
                "location": "unknown"
            },
            session_expires=datetime.fromtimestamp(payload["exp"])
        )
        
        # Update session tracking
        self.active_sessions[user.id] = {
            "last_activity": datetime.utcnow(),
            "token": token
        }
        
        return user
    
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        User login - returns token and user info
        MVP Note: In production, this would integrate with SAML 2.0 SSO
        """
        
        user = await self.authenticate_user(email, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Create access token
        access_token = self.create_access_token(user)
        
        # Log successful login
        logger.info("User login successful", 
                   user_id=user["id"], 
                   email=user["email"])
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_hours * 3600,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "roles": user["roles"],
                "is_admin": user["is_admin"]
            }
        }
    
    async def logout(self, user_id: str):
        """User logout - invalidate session"""
        
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            logger.info("User logout", user_id=user_id)
    
    def check_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has specific permission - MVP-FR-022
        """
        
        permissions = {
            "view_responses": lambda u: "user" in u.roles,
            "submit_queries": lambda u: "user" in u.roles,
            "upload_documents": lambda u: u.can_upload_documents,
            "view_audit_logs": lambda u: u.is_admin,
            "manage_users": lambda u: u.is_admin,
            "admin_access": lambda u: u.is_admin
        }
        
        permission_check = permissions.get(permission)
        if not permission_check:
            return False
        
        return permission_check(user)
    
    def require_permission(self, user: User, permission: str):
        """Require specific permission or raise exception"""
        
        if not self.check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        
        active_count = len(self.active_sessions)
        
        # Clean up expired sessions
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for user_id, session in self.active_sessions.items():
            # Session timeout after 8 hours of inactivity
            if (current_time - session["last_activity"]).total_seconds() > (8 * 3600):
                expired_sessions.append(user_id)
        
        for user_id in expired_sessions:
            del self.active_sessions[user_id]
        
        return {
            "active_sessions": len(self.active_sessions),
            "expired_sessions_cleaned": len(expired_sessions),
            "total_users": len(self.users_db)
        }
    
    # SAML 2.0 Integration Placeholder
    # In production, these methods would handle SAML authentication
    async def handle_saml_response(self, saml_response: str) -> User:
        """
        Handle SAML 2.0 authentication response
        This is a placeholder for production implementation
        """
        # TODO: Implement SAML 2.0 response parsing
        # TODO: Extract user attributes from SAML assertion
        # TODO: Map SAML attributes to internal user model
        # TODO: Create or update user record
        # TODO: Generate session token
        
        raise NotImplementedError("SAML 2.0 integration pending")
    
    async def initiate_saml_login(self, relay_state: Optional[str] = None) -> str:
        """
        Initiate SAML 2.0 login
        Returns URL to redirect user to IdP
        """
        # TODO: Generate SAML AuthnRequest
        # TODO: Redirect to IdP login URL
        
        raise NotImplementedError("SAML 2.0 integration pending")