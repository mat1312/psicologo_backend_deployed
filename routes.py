from typing import Dict, List, Optional
from fastapi import APIRouter, Depends
import logging
from main import recommend_resources, ResourceRequest, ResourceResponse, verify_token

# Configurazione logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/patients/{patient_id}/recommendations", response_model=ResourceResponse)
async def patient_recommendations_endpoint(
    patient_id: str, 
    query: Optional[str] = None, 
    user_data: Dict = Depends(verify_token)
):
    """Endpoint for patient-specific resource recommendations."""
    logger.info(f"Patient recommendations request for patient: {patient_id}, query: {query}")
    logger.info(f"User data from router: {user_data}")  # Log user data for debugging
    
    # Create a resource request with the session ID from the query parameter
    request = ResourceRequest(
        query=query or "", 
        session_id=query or ""  # Use query as session_id if provided
    )
    
    try:
        # Reuse the existing recommend_resources logic
        return await recommend_resources(request)
    except Exception as e:
        logger.error(f"Error in router recommendations: {str(e)}")
        # Return empty resources as fallback to avoid 403 errors
        return ResourceResponse(resources=[]) 