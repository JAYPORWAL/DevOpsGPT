import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from models.content import DevOpsContent
from models.tracking import QuizAttempt, UserProgress, TopicProgress
from utils.logger import setup_logger

logger = setup_logger("StorageService")

CORE_TOPICS = ["Docker", "Kubernetes", "Jenkins", "Terraform", "AWS", "Linux", "Git", "CI/CD"]

class StorageService:
    """
    Manages local JSON storage for topic caching, chat history, quiz attempts,
    and user learning progress.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache")
        self.chat_dir = os.path.join(data_dir, "chat")
        self.progress_dir = os.path.join(data_dir, "progress")
        self.uploads_dir = os.path.join(data_dir, "uploads")
        self.faiss_dir = os.path.join(data_dir, "faiss_index")
        
        # Ensure directories exist
        for directory in [self.cache_dir, self.chat_dir, self.progress_dir, self.uploads_dir, self.faiss_dir]:
            os.makedirs(directory, exist_ok=True)
            
        self.progress_file = os.path.join(self.progress_dir, "user_progress.json")
        self.quiz_file = os.path.join(self.progress_dir, "quiz_attempts.json")
        
    def _sanitize_filename(self, name: str) -> str:
        """Sanitizes strings to be safe for filenames."""
        return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in name]).lower()

    # --- TOPIC CACHE MANAGEMENT ---
    def save_to_cache(self, topic: str, content: DevOpsContent) -> bool:
        """Saves generated content into the cache directory."""
        filename = f"{self._sanitize_filename(topic)}.json"
        filepath = os.path.join(self.cache_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content.model_dump_json(indent=2))
            logger.info(f"Successfully cached topic content for: {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache topic {topic}: {e}", exc_info=True)
            return False

    def load_from_cache(self, topic: str) -> Optional[DevOpsContent]:
        """Loads and validates topic content from the cache, if it exists."""
        filename = f"{self._sanitize_filename(topic)}.json"
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            validated = DevOpsContent.model_validate(data)
            logger.info(f"Cache hit: Loaded topic content for: {topic}")
            return validated
        except Exception as e:
            logger.error(f"Cache corruption or validation error for {topic}: {e}", exc_info=True)
            return None

    def get_cached_topics(self) -> List[str]:
        """Returns a list of all topic names stored in the cache."""
        topics = []
        for file in os.listdir(self.cache_dir):
            if file.endswith(".json"):
                filepath = os.path.join(self.cache_dir, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "topic" in data:
                        topics.append(data["topic"])
                except Exception:
                    pass
        return list(set(topics))

    # --- QUIZ ATTEMPTS ---
    def save_quiz_attempt(self, attempt: QuizAttempt) -> bool:
        """Appends a new quiz attempt log to the quiz_attempts.json file."""
        try:
            attempts = self.load_quiz_attempts()
            attempts.append(attempt)
            
            with open(self.quiz_file, "w", encoding="utf-8") as f:
                json.dump([att.model_dump() for att in attempts], f, indent=2)
            
            # Update overall progress
            self.update_progress_on_quiz(attempt.topic, attempt.percentage)
            logger.info(f"Saved quiz attempt for {attempt.topic} with score {attempt.percentage}%")
            return True
        except Exception as e:
            logger.error(f"Failed to save quiz attempt: {e}", exc_info=True)
            return False

    def load_quiz_attempts(self) -> List[QuizAttempt]:
        """Loads and returns all quiz attempts."""
        if not os.path.exists(self.quiz_file):
            return []
        try:
            with open(self.quiz_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [QuizAttempt.model_validate(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load quiz attempts: {e}", exc_info=True)
            return []

    # --- USER PROGRESS TRACKING ---
    def load_progress(self) -> UserProgress:
        """Loads overall user learning progress."""
        if not os.path.exists(self.progress_file):
            # Pre-populate with core topics as uncompleted
            initial_topics = {}
            for topic in CORE_TOPICS:
                initial_topics[topic] = TopicProgress(
                    topic=topic,
                    completed=False,
                    last_studied=datetime.now().isoformat()
                )
            progress = UserProgress(topics=initial_topics)
            self.save_progress(progress)
            return progress
            
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserProgress.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load user progress: {e}", exc_info=True)
            return UserProgress()

    def save_progress(self, progress: UserProgress) -> bool:
        """Saves user progress state."""
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                f.write(progress.model_dump_json(indent=2))
            return True
        except Exception as e:
            logger.error(f"Failed to save user progress: {e}", exc_info=True)
            return False

    def update_progress_on_completion(self, topic: str, completed: bool = True) -> None:
        """Marks a topic as studied / completed."""
        progress = self.load_progress()
        if topic not in progress.topics:
            progress.topics[topic] = TopicProgress(
                topic=topic,
                completed=completed,
                last_studied=datetime.now().isoformat()
            )
        else:
            progress.topics[topic].completed = completed
            progress.topics[topic].last_studied = datetime.now().isoformat()
            
        self._recalculate_weak_topics(progress)
        self.save_progress(progress)

    def update_progress_on_quiz(self, topic: str, score_pct: float) -> None:
        """Updates quiz statistics for a topic."""
        progress = self.load_progress()
        if topic not in progress.topics:
            progress.topics[topic] = TopicProgress(
                topic=topic,
                completed=False,
                last_studied=datetime.now().isoformat(),
                quizzes_taken=1,
                best_score=score_pct
            )
        else:
            tp = progress.topics[topic]
            tp.quizzes_taken += 1
            if tp.best_score is None or score_pct > tp.best_score:
                tp.best_score = score_pct
            tp.last_studied = datetime.now().isoformat()
            
        self._recalculate_weak_topics(progress)
        self.save_progress(progress)

    def _recalculate_weak_topics(self, progress: UserProgress) -> None:
        """Identifies weak topics (average or best score below 70%)."""
        weak = []
        for topic, tp in progress.topics.items():
            if tp.best_score is not None and tp.best_score < 70.0:
                weak.append(topic)
            # If they attempted a quiz and score is low, or they haven't completed but attempted
        progress.weak_topics = list(set(weak))

    # --- CHAT HISTORY STORAGE ---
    def save_chat_message(self, session_id: str, role: str, content: str) -> None:
        """Saves a single message in a local session file."""
        filepath = os.path.join(self.chat_dir, f"{session_id}.json")
        history = self.load_chat_history(session_id)
        
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat message for session {session_id}: {e}")

    def load_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Loads conversational logs for a given session ID."""
        filepath = os.path.join(self.chat_dir, f"{session_id}.json")
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history for session {session_id}: {e}")
            return []

    def list_chat_sessions(self) -> List[str]:
        """Lists all existing chat session file IDs (sorted by modified date)."""
        files = [f for f in os.listdir(self.chat_dir) if f.endswith(".json")]
        # Sort by last modified time
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.chat_dir, x)), reverse=True)
        return [f.replace(".json", "") for f in files]
