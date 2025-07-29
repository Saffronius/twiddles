import pytest
from unittest.mock import Mock, AsyncMock
from app.services.router_service import route_scratchpad
from app.services.embedding_service import cosine_similarity
from app.models.models import Channel, ScratchpadEntry

def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Same vectors should have similarity 1.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 0.001

    # Orthogonal vectors should have similarity 0.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec1, vec2) - 0.0) < 0.001

    # Opposite vectors should have similarity -1.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec2) - (-1.0)) < 0.001

@pytest.mark.asyncio
async def test_router_suggests_new_when_no_channels():
    """Test router suggests new channel when no channels exist."""
    # Mock database session
    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = Mock(
        id="entry-1",
        embedding=[1.0, 0.0, 0.0],
        state='staged'
    )
    mock_db.query.return_value.filter.return_value.all.return_value = []

    result = await route_scratchpad(mock_db, "entry-1")
    assert result["suggest_new"] is True

@pytest.mark.asyncio 
async def test_router_suggests_new_when_below_threshold():
    """Test router suggests new channel when similarity below threshold."""
    # Mock database session with low similarity channel
    mock_db = Mock()
    
    # Mock scratchpad entry
    mock_entry = Mock()
    mock_entry.id = "entry-1"
    mock_entry.embedding = [1.0, 0.0, 0.0]
    mock_entry.state = 'staged'
    
    # Mock channel with orthogonal embedding (similarity = 0)
    mock_channel = Mock()
    mock_channel.id = "channel-1"
    mock_channel.embedding_centroid = [0.0, 1.0, 0.0]
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_channel]

    result = await route_scratchpad(mock_db, "entry-1")
    assert result["suggest_new"] is True

@pytest.mark.asyncio
async def test_router_suggests_channel_when_above_threshold():
    """Test router suggests existing channel when similarity above threshold.""" 
    # Mock database session with high similarity channel
    mock_db = Mock()
    
    # Mock scratchpad entry
    mock_entry = Mock()
    mock_entry.id = "entry-1" 
    mock_entry.embedding = [1.0, 0.0, 0.0]
    mock_entry.state = 'staged'
    
    # Mock channel with similar embedding (similarity = 1.0)
    mock_channel = Mock()
    mock_channel.id = "channel-1"
    mock_channel.embedding_centroid = [1.0, 0.0, 0.0]
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_channel]

    result = await route_scratchpad(mock_db, "entry-1")
    assert result["suggest_new"] is False
    assert result["best_channel_id"] == "channel-1"
    assert result["confidence"] == 1.0 