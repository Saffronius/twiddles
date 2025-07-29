"""Initial migration with pgvector

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create channels table
    op.create_table('channels',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('description_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('inherit_prompt', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('embedding_centroid', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.ForeignKeyConstraint(['parent_id'], ['channels.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_channels_parent_id'), 'channels', ['parent_id'], unique=False)
    
    # Create messages table
    op.create_table('messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('channel_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('memory_layer', sa.Text(), nullable=True),
        sa.Column('scratchpad_origin_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name='messages_role_check'),
        sa.CheckConstraint("memory_layer IN ('recent', 'compressed')", name='messages_memory_layer_check'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_channel_id_created_at'), 'messages', ['channel_id', 'created_at'], unique=False)
    
    # Create scratchpad_entries table
    op.create_table('scratchpad_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('content_text', sa.Text(), nullable=False),
        sa.Column('state', sa.Text(), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("state IN ('staged', 'routed')", name='scratchpad_entries_state_check'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create routing_logs table
    op.create_table('routing_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('entry_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_channel_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('routed_by', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['entry_id'], ['scratchpad_entries.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create memory_summaries table
    op.create_table('memory_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('channel_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_message_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create context_slices table
    op.create_table('context_slices',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('channel_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_message_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('included_message_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('included_summary_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column('token_estimate', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("mode IN ('plain', 'rag')", name='context_slices_mode_check'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ),
        sa.ForeignKeyConstraint(['user_message_id'], ['messages.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create knowledge_cards table
    op.create_table('knowledge_cards',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('channel_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('source_message_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Update embedding columns to use pgvector type
    op.execute('ALTER TABLE channels ALTER COLUMN embedding_centroid TYPE vector(1536) USING embedding_centroid::vector(1536)')
    op.execute('ALTER TABLE messages ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)')
    op.execute('ALTER TABLE scratchpad_entries ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)')
    
    # Create vector indexes
    op.execute('CREATE INDEX ix_messages_embedding ON messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')


def downgrade() -> None:
    op.drop_table('knowledge_cards')
    op.drop_table('context_slices')
    op.drop_table('memory_summaries')
    op.drop_table('routing_logs')
    op.drop_table('scratchpad_entries')
    op.drop_table('messages')
    op.drop_table('channels')
    op.execute('DROP EXTENSION IF EXISTS vector') 