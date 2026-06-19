import os
import sys

# Ensure the root is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.storage_service import StorageService, CORE_TOPICS
from services.gemini_service import GeminiService

def prepopulate():
    print("Pre-populating local cache for core DevOps topics...")
    storage_svc = StorageService(data_dir=os.path.join(project_root, "data"))
    gemini_svc = GeminiService()  # Offline capability does not require API key
    
    for topic in CORE_TOPICS:
        # Generate offline structured content
        content = gemini_svc._generate_offline_devops_content(topic)
        # Save to local cache
        success = storage_svc.save_to_cache(topic, content)
        if success:
            print(f"  - Successfully cached content for topic: {topic}")
        else:
            print(f"  - Failed to cache content for topic: {topic}")
            
    print("Pre-population complete!")

if __name__ == "__main__":
    prepopulate()
