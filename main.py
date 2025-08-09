#!/usr/bin/env python3

import os
import sys
from typing import Optional
from app.model_handler import ModelHandler
from app.retriever import RAGRetriever
from app.chat_manager import ChatManager


class RAGChatbot:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.retriever = RAGRetriever()
        self.chat_manager = ChatManager()
        self.current_user_id = None
        self.current_session_id = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        print("Initializing RAG Chatbot...")
        
        print("Loading language model...")
        if not self.model_handler.load_model():
            print("Failed to load language model")
            return False
        
        print("Initializing RAG retriever...")
        if not self.retriever.initialize():
            print("No existing knowledge base found. Loading documents...")
            if not self.retriever.load_documents_from_directory():
                print("Warning: No documents loaded in knowledge base")
        
        doc_count = self.retriever.get_document_count()
        print(f"Knowledge base loaded with {doc_count} documents")
        
        self.is_initialized = True
        print("RAG Chatbot initialized successfully!")
        return True
    
    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        self.current_user_id = user_id
        self.current_session_id = self.chat_manager.create_session(user_id, session_id)
        print(f"Started session {self.current_session_id} for user {user_id}")
        return self.current_session_id
    
    def process_query(self, query: str, use_rag: bool = True) -> str:
        if not self.is_initialized:
            return "Chatbot not initialized. Please run initialize() first."
        
        if not self.current_user_id or not self.current_session_id:
            return "No active session. Please start a session first."
        
        try:
            self.chat_manager.add_message(
                self.current_user_id, 
                self.current_session_id, 
                "user", 
                query
            )
            
            if use_rag and self.retriever.get_document_count() > 0:
                context = self.retriever.get_context_for_query(query)
                if context:
                    response = self.model_handler.generate_with_context(context, query)
                else:
                    response = self.model_handler.generate_response(query)
            else:
                conversation_context = self.chat_manager.get_context_from_history(
                    self.current_user_id, 
                    self.current_session_id, 
                    max_context_messages=3
                )
                
                if conversation_context:
                    enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
                    response = self.model_handler.generate_response(enhanced_query)
                else:
                    response = self.model_handler.generate_response(query)
            
            self.chat_manager.add_message(
                self.current_user_id, 
                self.current_session_id, 
                "assistant", 
                response
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return "I apologize, but I encountered an error processing your request."
    
    def get_session_history(self) -> list:
        if not self.current_user_id or not self.current_session_id:
            return []
        
        return self.chat_manager.get_conversation_history(
            self.current_user_id, 
            self.current_session_id
        )
    
    def interactive_mode(self):
        if not self.initialize():
            print("Failed to initialize chatbot")
            return
        
        print("\n=== RAG Chatbot Interactive Mode ===")
        print("Commands:")
        print("  /user <user_id> - Set user ID")
        print("  /session [session_id] - Start new session or specify session ID")
        print("  /history - Show conversation history")
        print("  /sessions - List user sessions")
        print("  /info - Show current session info")
        print("  /quit - Exit")
        print("\nType your questions or use commands...")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/quit'):
                    print("Goodbye!")
                    break
                
                elif user_input.startswith('/user'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        user_id = parts[1].strip()
                        self.current_user_id = user_id
                        print(f"User set to: {user_id}")
                    else:
                        print("Usage: /user <user_id>")
                
                elif user_input.startswith('/session'):
                    if not self.current_user_id:
                        print("Please set user ID first with /user <user_id>")
                        continue
                    
                    parts = user_input.split(' ', 1)
                    session_id = parts[1].strip() if len(parts) > 1 else None
                    self.start_session(self.current_user_id, session_id)
                
                elif user_input.startswith('/history'):
                    history = self.get_session_history()
                    if history:
                        print("\nConversation History:")
                        for msg in history:
                            role = msg['role'].capitalize()
                            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            print(f"{role}: {content}")
                    else:
                        print("No conversation history found")
                
                elif user_input.startswith('/sessions'):
                    if not self.current_user_id:
                        print("Please set user ID first")
                        continue
                    
                    sessions = self.chat_manager.get_user_sessions(self.current_user_id)
                    if sessions:
                        print(f"Sessions for user {self.current_user_id}:")
                        for session in sessions:
                            info = self.chat_manager.get_session_info(self.current_user_id, session)
                            print(f"  {session}: {info['message_count']} messages")
                    else:
                        print("No sessions found")
                
                elif user_input.startswith('/info'):
                    if self.current_user_id and self.current_session_id:
                        info = self.chat_manager.get_session_info(self.current_user_id, self.current_session_id)
                        print(f"Current session: {info}")
                        print(f"Documents in knowledge base: {self.retriever.get_document_count()}")
                    else:
                        print("No active session")
                
                else:
                    if not self.current_user_id:
                        print("Please set user ID first with /user <user_id>")
                        continue
                    
                    if not self.current_session_id:
                        self.start_session(self.current_user_id)
                    
                    response = self.process_query(user_input)
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        chatbot = RAGChatbot()
        chatbot.interactive_mode()
    else:
        chatbot = RAGChatbot()
        if chatbot.initialize():
            chatbot.start_session("demo_user")
            
            test_queries = [
                "What is machine learning?",
                "How does neural networks work?",
                "Tell me about artificial intelligence"
            ]
            
            print("\n=== Demo Queries ===")
            for query in test_queries:
                print(f"\nUser: {query}")
                response = chatbot.process_query(query)
                print(f"Assistant: {response}")


if __name__ == "__main__":
    main()