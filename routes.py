from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, Header, HTTPException
import logging

# Configurazione logging
logger = logging.getLogger(__name__)

# Create router without imports from main
router = APIRouter()

# Define these here instead of importing from main
class ResourceRequest:
    def __init__(self, query: str, session_id: str):
        self.query = query
        self.session_id = session_id

class ResourceResponse:
    def __init__(self, resources: List[Dict[str, str]]):
        self.resources = resources

# Reference to recommend_resources - will be set by main.py
recommend_resources_func = None
verify_token_func = None

@router.get("/patients/{patient_id}/recommendations")
async def patient_recommendations_endpoint(
    patient_id: str, 
    query: Optional[str] = None, 
    authorization: str = Header(None)
):
    """Endpoint for patient-specific resource recommendations."""
    if not verify_token_func or not recommend_resources_func:
        logger.error("Functions not initialized")
        return {"resources": []}
    
    try:
        # Verify token
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            user_data = await verify_token_func(token)
        else:
            # For development, use a demo user
            user_data = {
                "id": "demo-user-id",
                "email": "demo@example.com",
                "role": "patient"
            }
        
        logger.info(f"Patient recommendations request for patient: {patient_id}, query: {query}")
        
        # Create a resource request with the session ID from the query parameter
        request = ResourceRequest(
            query=query or "", 
            session_id=query or ""  # Use query as session_id if provided
        )
        
        # Reuse the existing recommend_resources logic
        result = await recommend_resources_func(request)
        return result
    except Exception as e:
        logger.error(f"Error in router recommendations: {str(e)}")
        # Return empty resources as fallback to avoid 403 errors
        return {"resources": []}

# Function to initialize the router with dependencies
def init_router(recommend_resources_function, verify_token_function):
    global recommend_resources_func, verify_token_func
    recommend_resources_func = recommend_resources_function
    verify_token_func = verify_token_function
    return router 