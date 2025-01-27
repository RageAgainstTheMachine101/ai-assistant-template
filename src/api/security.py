from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from enum import Enum
import re
from .database.supabase_client import SupabaseManager

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class SecurityConfig:
    SUSPICIOUS_PATTERNS = [
        r"system\s*prompt",
        r"hidden\s*instructions",
        r"base\s*prompt",
        r"original\s*prompt",
        r"underlying\s*instructions",
        r"prompt\s*injection"
    ]

    @staticmethod
    async def validate_api_key(api_key: Optional[str] = Security(api_key_header)) -> tuple[str, UserRole]:
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key is missing"
            )
        
        # Validate API key against Supabase
        supabase = SupabaseManager()
        result = await supabase.validate_api_key(api_key)
        
        if not result:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
            
        return api_key, result.get('role', UserRole.GUEST)

    @staticmethod
    def check_prompt_injection(query: str) -> bool:
        """Check if the query contains suspicious patterns that might indicate prompt injection attempts"""
        query = query.lower()
        return any(re.search(pattern, query) for pattern in SecurityConfig.SUSPICIOUS_PATTERNS)

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize the query to prevent prompt injection"""
        if SecurityConfig.check_prompt_injection(query):
            raise HTTPException(
                status_code=400,
                detail="Query contains suspicious patterns and has been rejected"
            )
        return query
