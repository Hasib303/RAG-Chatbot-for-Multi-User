import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class ChatManager:
    def __init__(self, chat_history_path: str = "data/chat_history"):
        self.chat_history_path = chat_history_path
        os.makedirs(self.chat_history_path, exist_ok=True)
    
    def _get_user_file_path(self, user_id: str) -> str:
        return os.path.join(self.chat_history_path, f"user_{user_id}.json")
    
    def _load_user_data(self, user_id: str) -> Dict:
        file_path = self._get_user_file_path(user_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user data for {user_id}: {str(e)}")
                return {"sessions": {}}
        return {"sessions": {}}
    
    def _save_user_data(self, user_id: str, data: Dict):
        file_path = self._get_user_file_path(user_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving user data for {user_id}: {str(e)}")
    
    def create_session(self, user_id: str, session_id: str = None) -> str:
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        user_data = self._load_user_data(user_id)
        
        if session_id not in user_data["sessions"]:
            user_data["sessions"][session_id] = {
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
            self._save_user_data(user_id, user_data)
        
        return session_id
    
    def add_message(self, user_id: str, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        user_data = self._load_user_data(user_id)
        
        if session_id not in user_data["sessions"]:
            user_data["sessions"][session_id] = {
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
        
        user_data["sessions"][session_id]["messages"].append(message)
        self._save_user_data(user_id, user_data)
    
    def get_conversation_history(self, user_id: str, session_id: str, max_messages: int = 10) -> List[Dict]:
        user_data = self._load_user_data(user_id)
        
        if session_id in user_data["sessions"]:
            messages = user_data["sessions"][session_id]["messages"]
            return messages[-max_messages:] if max_messages > 0 else messages
        
        return []
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        user_data = self._load_user_data(user_id)
        return list(user_data["sessions"].keys())
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        user_data = self._load_user_data(user_id)
        
        if session_id in user_data["sessions"]:
            del user_data["sessions"][session_id]
            self._save_user_data(user_id, user_data)
            return True
        
        return False
    
    def get_context_from_history(self, user_id: str, session_id: str, max_context_messages: int = 5) -> str:
        history = self.get_conversation_history(user_id, session_id, max_context_messages)
        
        if not history:
            return ""
        
        context_parts = []
        for message in history:
            role = message["role"]
            content = message["content"]
            context_parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_all_sessions(self, user_id: str):
        user_data = {"sessions": {}}
        self._save_user_data(user_id, user_data)
    
    def get_session_info(self, user_id: str, session_id: str) -> Optional[Dict]:
        user_data = self._load_user_data(user_id)
        
        if session_id in user_data["sessions"]:
            session_data = user_data["sessions"][session_id]
            return {
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "message_count": len(session_data["messages"]),
                "last_message": session_data["messages"][-1]["timestamp"] if session_data["messages"] else None
            }
        
        return None