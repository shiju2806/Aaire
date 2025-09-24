"""
Intelligent Conversation Memory Manager for AAIRE
Level 2: Intelligent Context Management with semantic compression and entity tracking
"""

import json
import redis
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import structlog
import re

logger = structlog.get_logger()

@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    entities: List[str] = None
    importance_score: float = 1.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []

@dataclass
class ConversationSummary:
    """Compressed summary of conversation segments"""
    topics: List[str]
    key_entities: List[str]
    important_facts: List[str]
    timespan: str
    original_messages: int

class ActuarialEntityExtractor:
    """Extracts important actuarial/insurance entities from text"""

    def __init__(self, config: Dict = None):
        """Initialize entity extractor with configuration"""
        self.config = config or {}

    # Key patterns for actuarial domain
    ACTUARIAL_PATTERNS = {
        'policy_types': r'\b(universal life|UL|term life|whole life|variable life|VL|ULSG)\b',
        'percentages': r'\b\d+(?:\.\d+)?%\b',
        'years': r'\b(?:year|years?)\s+\d+(?:-\d+)?\b|\b\d+(?:-\d+)?\s+(?:year|years?)\b',
        'sections': r'\bSection\s+\d+(?:\.[A-Z0-9.]+)?\b',
        'calculations': r'\b(reserve|premium|calculation|NPR|adjusted gross premium)\b',
        'regulations': r'\b(LDTI|GAAP|SAP|VM-\d+|Valuation Manual)\b',
        'monetary': r'\$[\d,]+(?:\.\d{2})?(?:\s+(?:million|billion|thousand))?',
        'formulas': r'\b[A-Z](?:_\d+)?(?:\s*[=+\-*/]\s*[A-Z](?:_\d+)?)*\b'
    }
    
    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        """Extract important entities from text"""
        entities = []
        
        for category, pattern in cls.ACTUARIAL_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    entities.append(f"{category}:{match}")
                elif isinstance(match, tuple):
                    entities.append(f"{category}:{match[0]}")
        
        return list(set(entities))  # Remove duplicates
    
    def calculate_importance(self, text: str) -> float:
        """Calculate importance score based on content using configurable weights"""
        # Get importance scoring config with defaults
        importance_config = self.config.get('importance_scoring', {})

        base_importance = importance_config.get('base_importance', 1.0)
        question_bonus = importance_config.get('question_bonus', 0.3)
        calculation_bonus = importance_config.get('calculation_bonus', 0.4)
        number_bonus = importance_config.get('number_bonus', 0.3)
        regulatory_bonus = importance_config.get('regulatory_bonus', 0.2)
        short_message_penalty = importance_config.get('short_message_penalty', 0.8)
        short_message_threshold = importance_config.get('short_message_threshold', 50)
        max_importance = importance_config.get('max_importance', 2.0)

        importance = base_importance

        # Higher importance for questions
        if '?' in text:
            importance += question_bonus

        # Higher importance for calculations/formulas
        if any(word in text.lower() for word in ['calculate', 'formula', 'equation']):
            importance += calculation_bonus

        # Higher importance for specific numbers
        if re.search(r'\d+(?:\.\d+)?%', text):
            importance += number_bonus

        # Higher importance for regulatory references
        if re.search(r'\b(Section|VM-|LDTI|GAAP)\b', text, re.IGNORECASE):
            importance += regulatory_bonus

        # Lower importance for very short messages
        if len(text) < short_message_threshold:
            importance *= short_message_penalty

        return min(importance, max_importance)

class ConversationMemoryManager:
    """Manages intelligent conversation memory with compression and entity tracking"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, config: Dict = None):
        self.config = config or {}
        self.redis_client = redis_client
        self.entity_extractor = ActuarialEntityExtractor(self.config)
        
        # Configuration settings
        self.max_messages_before_compression = self.config.get('max_messages_before_compression', 15)
        self.session_ttl_hours = self.config.get('session_ttl_hours', 2)
        self.max_compressed_summaries = self.config.get('max_compressed_summaries', 3)
        self.compression_ratio = self.config.get('compression_ratio', 0.3)
        
        logger.info("ConversationMemoryManager initialized", 
                   max_messages=self.max_messages_before_compression,
                   ttl_hours=self.session_ttl_hours)
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session"""
        return f"aaire:conversation:{session_id}"
    
    def _get_summary_key(self, session_id: str) -> str:
        """Get Redis key for conversation summaries"""
        return f"aaire:conversation:summary:{session_id}"
    
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a new message to conversation memory"""
        try:
            # Extract entities and calculate importance
            entities = self.entity_extractor.extract_entities(content)
            importance = self.entity_extractor.calculate_importance(content)
            
            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
                entities=entities,
                importance_score=importance
            )
            
            # Store in Redis if available, otherwise in memory
            if self.redis_client:
                await self._store_message_redis(session_id, message)
            
            # Check if compression is needed
            messages = await self.get_messages(session_id)
            if len(messages) >= self.max_messages_before_compression:
                await self._compress_old_messages(session_id, messages)
                
            logger.debug("Message added to conversation memory", 
                        session_id=session_id[:8], 
                        role=role, 
                        entities_count=len(entities),
                        importance=importance)
                        
        except Exception as e:
            logger.error("Failed to add message to conversation memory",
                        exception_details=str(e), session_id=session_id[:8])
    
    async def _store_message_redis(self, session_id: str, message: ConversationMessage) -> None:
        """Store message in Redis"""
        key = self._get_session_key(session_id)
        message_data = {
            'role': message.role,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'entities': message.entities,
            'importance_score': message.importance_score
        }
        
        # Add to list (most recent first)
        self.redis_client.lpush(key, json.dumps(message_data))
        
        # Set TTL
        ttl_seconds = int(self.session_ttl_hours * 3600)
        self.redis_client.expire(key, ttl_seconds)
    
    async def get_messages(self, session_id: str, limit: int = None) -> List[ConversationMessage]:
        """Get conversation messages for session"""
        try:
            if self.redis_client:
                return await self._get_messages_redis(session_id, limit)
            else:
                return []  # Fallback to empty if no storage
                
        except Exception as e:
            logger.error("Failed to retrieve conversation messages",
                        exception_details=str(e), session_id=session_id[:8])
            return []
    
    async def _get_messages_redis(self, session_id: str, limit: int = None) -> List[ConversationMessage]:
        """Get messages from Redis"""
        key = self._get_session_key(session_id)
        
        # Get messages (Redis returns most recent first due to lpush)
        if limit:
            raw_messages = self.redis_client.lrange(key, 0, limit - 1)
        else:
            raw_messages = self.redis_client.lrange(key, 0, -1)
        
        messages = []
        for raw_msg in raw_messages:
            try:
                msg_data = json.loads(raw_msg)
                message = ConversationMessage(
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=datetime.fromisoformat(msg_data['timestamp']),
                    entities=msg_data.get('entities', []),
                    importance_score=msg_data.get('importance_score', 1.0)
                )
                messages.append(message)
            except Exception as e:
                logger.warning("Failed to parse message", exception_details=str(e))
        
        # Reverse to get chronological order (oldest first)
        return list(reversed(messages))
    
    async def _compress_old_messages(self, session_id: str, messages: List[ConversationMessage]) -> None:
        """Compress old messages into summaries"""
        try:
            # Split messages into recent (keep) and old (compress)
            split_point = int(len(messages) * self.compression_ratio)
            messages_to_compress = messages[:split_point]
            messages_to_keep = messages[split_point:]
            
            if not messages_to_compress:
                return
            
            # Create summary of old messages
            summary = await self._create_summary(messages_to_compress)
            
            # Store summary
            await self._store_summary(session_id, summary)
            
            # Remove old messages and keep only recent ones
            if self.redis_client:
                key = self._get_session_key(session_id)
                # Clear the list
                self.redis_client.delete(key)
                # Re-add recent messages
                for message in reversed(messages_to_keep):  # Reverse for lpush
                    await self._store_message_redis(session_id, message)
            
            logger.info("Compressed conversation messages", 
                       session_id=session_id[:8],
                       compressed=len(messages_to_compress),
                       kept=len(messages_to_keep))
                       
        except Exception as e:
            logger.error("Failed to compress conversation messages",
                        exception_details=str(e), session_id=session_id[:8])
    
    async def _create_summary(self, messages: List[ConversationMessage]) -> ConversationSummary:
        """Create a compressed summary of messages"""
        # Extract all entities and topics
        all_entities = []
        topics = set()
        important_facts = []
        
        for message in messages:
            all_entities.extend(message.entities)
            
            # Extract topics from high-importance messages
            if message.importance_score > 1.2:
                # Simple topic extraction - look for key phrases
                content_lower = message.content.lower()
                if 'reserve' in content_lower:
                    topics.add('reserves')
                if 'premium' in content_lower:
                    topics.add('premiums')
                if 'universal life' in content_lower or 'ul' in content_lower:
                    topics.add('universal_life')
                if 'calculation' in content_lower:
                    topics.add('calculations')
                
                # Store important facts (questions and key statements)
                if '?' in message.content or message.importance_score > 1.5:
                    fact = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    important_facts.append(fact)
        
        # Get unique entities and their frequency
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Keep most frequent entities
        key_entities = sorted(entity_counts.keys(), 
                             key=lambda x: entity_counts[x], 
                             reverse=True)[:10]
        
        # Create timespan description
        if messages:
            start_time = messages[0].timestamp
            end_time = messages[-1].timestamp
            duration = end_time - start_time
            timespan = f"{duration.total_seconds() / 60:.1f} minutes"
        else:
            timespan = "0 minutes"
        
        return ConversationSummary(
            topics=list(topics),
            key_entities=key_entities,
            important_facts=important_facts[:5],  # Keep top 5 facts
            timespan=timespan,
            original_messages=len(messages)
        )
    
    async def _store_summary(self, session_id: str, summary: ConversationSummary) -> None:
        """Store conversation summary"""
        if not self.redis_client:
            return
            
        key = self._get_summary_key(session_id)
        summary_data = asdict(summary)
        
        # Add timestamp
        summary_data['created_at'] = datetime.utcnow().isoformat()
        
        # Store as list item (keep multiple summaries)
        self.redis_client.lpush(key, json.dumps(summary_data))
        
        # Limit number of summaries
        self.redis_client.ltrim(key, 0, self.max_compressed_summaries - 1)
        
        # Set TTL
        ttl_seconds = int(self.session_ttl_hours * 3600)
        self.redis_client.expire(key, ttl_seconds)
    
    async def get_conversation_context(self, session_id: str, max_tokens: int = 2000) -> str:
        """Get intelligent conversation context for LLM prompt"""
        try:
            # Get recent messages
            recent_messages = await self.get_messages(session_id, limit=10)
            
            # Get summaries
            summaries = await self._get_summaries(session_id)
            
            # Build context string
            context_parts = []
            
            # Add summaries first (older context)
            if summaries:
                context_parts.append("## Previous Conversation Context:")
                for i, summary in enumerate(reversed(summaries)):  # Oldest first
                    summary_text = self._format_summary(summary, i + 1)
                    context_parts.append(summary_text)
            
            # Add recent messages
            if recent_messages:
                context_parts.append("\n## Recent Conversation:")
                for message in recent_messages:
                    role = "User" if message.role == 'user' else "Assistant"
                    content = message.content[:300] + "..." if len(message.content) > 300 else message.content
                    context_parts.append(f"{role}: {content}")
            
            full_context = "\n".join(context_parts)
            
            # Truncate if too long (rough token estimation: ~4 chars per token)
            if len(full_context) > max_tokens * 4:
                full_context = full_context[:max_tokens * 4 - 100] + "\n... (context truncated)"
            
            return full_context
            
        except Exception as e:
            logger.error("Failed to get conversation context",
                        exception_details=str(e), session_id=session_id[:8])
            return ""
    
    def _format_summary(self, summary_data: Dict, summary_num: int) -> str:
        """Format summary for context"""
        topics = ", ".join(summary_data.get('topics', []))
        entities = ", ".join(summary_data.get('key_entities', [])[:5])  # Top 5 entities
        facts = summary_data.get('important_facts', [])
        
        formatted = f"Summary {summary_num} ({summary_data.get('timespan', 'unknown duration')}):\n"
        if topics:
            formatted += f"- Topics discussed: {topics}\n"
        if entities:
            formatted += f"- Key elements: {entities}\n"
        if facts:
            formatted += f"- Important points: {'; '.join(facts[:2])}\n"  # Top 2 facts
        
        return formatted
    
    async def _get_summaries(self, session_id: str) -> List[Dict]:
        """Get conversation summaries"""
        if not self.redis_client:
            return []
            
        try:
            key = self._get_summary_key(session_id)
            raw_summaries = self.redis_client.lrange(key, 0, -1)
            
            summaries = []
            for raw_summary in raw_summaries:
                try:
                    summary_data = json.loads(raw_summary)
                    summaries.append(summary_data)
                except Exception as e:
                    logger.warning("Failed to parse summary", exception_details=str(e))
            
            return summaries
            
        except Exception as e:
            logger.error("Failed to get summaries", exception_details=str(e), session_id=session_id[:8])
            return []
    
    async def clear_session(self, session_id: str) -> None:
        """Clear all memory for a session"""
        if self.redis_client:
            message_key = self._get_session_key(session_id)
            summary_key = self._get_summary_key(session_id)
            
            self.redis_client.delete(message_key)
            self.redis_client.delete(summary_key)
            
            logger.info("Cleared conversation memory", session_id=session_id[:8])
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a conversation session"""
        try:
            messages = await self.get_messages(session_id)
            summaries = await self._get_summaries(session_id)
            
            # Calculate stats
            total_messages = len(messages)
            total_entities = []
            avg_importance = 0
            
            if messages:
                for msg in messages:
                    total_entities.extend(msg.entities)
                avg_importance = sum(msg.importance_score for msg in messages) / len(messages)
            
            return {
                'session_id': session_id,
                'active_messages': total_messages,
                'compressed_summaries': len(summaries),
                'unique_entities': len(set(total_entities)),
                'avg_importance_score': round(avg_importance, 2),
                'memory_active': total_messages > 0 or len(summaries) > 0
            }
            
        except Exception as e:
            logger.error("Failed to get session stats", exception_details=str(e), session_id=session_id[:8])
            return {'session_id': session_id, 'error': str(e)}