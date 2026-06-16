from pydantic import BaseModel, Field
from typing import List

class RoadmapLevel(BaseModel):
    duration: str = Field(description="Estimated duration or timeline, e.g., '1 week'")
    topics: List[str] = Field(description="List of key concepts or subtopics to cover")

class Roadmap(BaseModel):
    beginner: RoadmapLevel = Field(description="Roadmap content for beginners")
    intermediate: RoadmapLevel = Field(description="Roadmap content for intermediate learners")
    advanced: RoadmapLevel = Field(description="Roadmap content for advanced learners")

class InterviewQuestion(BaseModel):
    id: int = Field(description="Question identifier")
    level: str = Field(description="Difficulty level: beginner, intermediate, or advanced")
    question: str = Field(description="DevOps interview question")
    answer: str = Field(description="Concise, ideal model answer (approx 2 sentences)")
    common_mistakes: List[str] = Field(description="Common mistakes or pitfalls candidate might make")

class HandsOnTask(BaseModel):
    title: str = Field(description="Title of the lab/exercise")
    instructions: str = Field(description="Step-by-step description of what to do")
    commands: List[str] = Field(description="Terminal commands (Linux, Docker, K8s, Terraform) to run")

class DevOpsProject(BaseModel):
    title: str = Field(description="Project name")
    difficulty: str = Field(description="Difficulty level: Beginner, Intermediate, or Advanced")
    architecture: str = Field(description="High-level architecture description or overview")
    tech_stack: List[str] = Field(description="List of key technologies used (e.g. AWS, Jenkins, Ansible)")
    learning_outcomes: List[str] = Field(description="Primary skills or concepts learned from this project")

class QuizQuestion(BaseModel):
    id: int = Field(description="Question identifier from 1 to 10")
    question: str = Field(description="The multiple choice question text")
    options: List[str] = Field(description="List of exactly 4 options")
    correct_option: str = Field(description="The exact text of the correct option, must match one of the options")
    explanation: str = Field(description="Detailed explanation of why the correct option is correct")

class DevOpsContent(BaseModel):
    topic: str = Field(description="The DevOps topic studied (e.g. Terraform)")
    explanation: str = Field(description="A beginner-friendly overview with real-world use cases and system architecture diagram in Mermaid syntax")
    roadmap: Roadmap = Field(description="Structured learning roadmap")
    interview_questions: List[InterviewQuestion] = Field(description="Exactly 13 interview questions: 5 beginner, 5 intermediate, 3 advanced")
    hands_on_tasks: List[HandsOnTask] = Field(description="Practical hands-on exercises")
    projects: List[DevOpsProject] = Field(description="Exactly 3 projects (1 Beginner, 1 Intermediate, 1 Advanced)")
    quiz: List[QuizQuestion] = Field(description="Exactly 10 multiple choice questions")
