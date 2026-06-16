from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QuizAttempt(BaseModel):
    id: str = Field(description="Unique identifier for the attempt (typically timestamp)")
    topic: str = Field(description="The topic of the quiz")
    timestamp: str = Field(description="ISO timestamp of when the quiz was taken")
    score: int = Field(description="Number of correct answers")
    total_questions: int = Field(description="Total number of questions in the quiz")
    percentage: float = Field(description="Percentage score obtained")
    user_answers: Dict[str, str] = Field(description="Map of question_id (as string) to the option chosen by the user")

class TopicProgress(BaseModel):
    topic: str = Field(description="The DevOps topic name")
    completed: bool = Field(default=False, description="Whether the topic is marked as fully studied")
    last_studied: str = Field(description="ISO timestamp of last activity")
    quizzes_taken: int = Field(default=0, description="Number of times quizzes for this topic were taken")
    best_score: Optional[float] = Field(default=None, description="Highest percentage scored on this topic's quizzes")

class UserProgress(BaseModel):
    topics: Dict[str, TopicProgress] = Field(default_factory=dict, description="Map of topic_name -> TopicProgress")
    weak_topics: List[str] = Field(default_factory=list, description="Topics where average score is below a threshold")
