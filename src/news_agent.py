import os
import sys
in_pydantic_v2 = True
from typing import List, Dict
import pandas as pd
import streamlit as st
from datetime import datetime,timedelta
import requests
from newsapi import NewsApiClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import NEWS_API_KEY, HF_MODEL_NAME

class NewsEmbeddingAgent:
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.model_name = HF_MODEL_NAME

        self.newsapi = None
        if self.news_api_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.news_api_key)
            except:
                pass
        
        self.llm_model = None
        self.llm_tokenizer = None
        self.__initialize_llm()

        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            self.embedding_model = None

        self.news_embeddings = []
        self.news_documents = []
        self.news_metadata = []
        
    def __initialize_llm(self):
        try:
            if 'flan-t5' in self.model_name.lower():
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name,torch_dtype=torch.float32, low_cpu_mem_usage=True)
                self.model_type = 'seq2seq'
            elif 'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower():
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(self.model_name,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, low_cpu_mem_usage=True, device_map='auto' if torch.cuda.is_available() else None)
                self.model_type = 'causal'
            else:
                self.model_name = 'google/flan-t5-base'
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model_type = 'seq2seq'
            if torch.cuda.is_available() and self.model_type == 'seq2seq':
                self.llm_model = self.llm_model.to('cuda')
        
        except Exception as e:
            st.error(f"Error loading LLM model: {e}")
            self.llm_model = None
            self.llm_tokenizer = None
            self.model_type = None
    def fetch_news(self, query: str = 'stock market India', days: int =7) -> List[Dict]:
        articles = []
        if not self.newsapi:
            return self._get_dummy_news()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            response = self.newsapi.get_everything(q=query, from_param=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'), language='en', sort_by='relevancy', page_size=50)

            if response['status'] == 'ok':
                articles = response['articles']
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return self._get_dummy_news()
        return articles
    
    def _get_dummy_news(self) -> List[Dict]:
        return [
            {
                'title': 'Stock Market Hits New Highs Amid Economic Recovery',
                'description': 'The stock market continues to reach new highs as the economy shows signs of recovery post-pandemic.',
                'url': 'https://example.com/news1',
                'publishedAt': '2024-06-01T10:00:00Z'
            },
            {
                'title': 'Tech Stocks Lead the Rally in the Stock Market',
                'description': 'Technology stocks are leading the rally in the stock market, driven by strong earnings reports.',
                'url': 'https://example.com/news2',
                'publishedAt': '2024-06-02T12:00:00Z'
            },
            {
                'title': 'Global Markets React to Geopolitical Tensions',
                'description': 'Global markets are reacting to ongoing geopolitical tensions, with increased volatility expected.',
                'url': 'https://example.com/news3',
                'publishedAt': '2024-06-03T14:00:00Z'
            }
        ]
    def create_embeddings(self, articles: List[str]) -> bool:
        if not self.embedding_model:
            return False
        try:
            self.news_embeddings = []
            self.news_documents = []
            self.news_metadata = []

            documents=[]
            metadatas =[]
            ids=[]

            for article in articles:
                text = f'{article.get("title","")} - {article.get("description","")} - {article.get("content","")}'
                if text.strip():
                    documents.append(text)
                    metadatas.append({'title': article.get('title', ''),'source': article.get('source', {}).get('name',''),'published': article.get('publishedAt', '')})
            if documents:
                embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
                self.news_embeddings = embeddings
                self.news_documents = documents
                self.news_metadata = metadatas
                return True
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
        return False

    def query_news(self, query: str, n_results: int = 3) -> List[Dict]:
        if not self.embedding_model:
            return []
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

            similarities = cosine_similarity(query_embedding, self.news_embeddings)[0]

            top_indices = np.argsort(similarities)[::-1][:n_results]
            article_results = []

            for idx in top_indices:
                
                article_results.append({
                        'content':self.news_documents[idx],
                        'title': self.news_metadata[idx].get('title',''),
                        'source': self.news_metadata[idx].get('source',''),
                        'published': self.news_metadata[idx].get('published',''),
                        'similarity': similarities[idx]
                    })
            return article_results
        except Exception as e:
            st.error(f"Error querying news: {e}")
            return []
    def chat_completion(self, user_message: str, var_context: str ="", news_context: str ="") -> str:
        if not self.llm_model or not self.llm_tokenizer:
            return self._get_fallback_response(user_message, var_context)
        try:
            
            system_context = f"""ou are a financial risk analyst assistant specialising in Value at Risk (VaR) predictions.
            Help users understand VaR calculations, market risks, and provide insights based on current market news.

            Context about current VaR calculations:
            {var_context}

            Recent relevant market news 
            {news_context}

            Provide clear, concise and actionale insights based on the above information.
            """

            full_prompt = f"{system_context}\n\nUser Message: {user_message}\n\nAssistant:"
            if self.model_type == 'seq2seq':
                inputs = self.llm_tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {key: val.to('cuda') for key, val in inputs.items()}
                outputs = self.llm_model.generate(**inputs, max_length=300,num_beams=4, early_stopping=True, temperature=0.7, top_p=0.9, do_sample=True)
            
                response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif self.model_type == 'causal':
                inputs = self.llm_tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=1024)
                if torch.cuda.is_available():
                    inputs = {key: val.to('cuda') for key, val in inputs.items()}
                outputs = self.llm_model.generate(**inputs, max_length=300, temperature=0.7, top_p=0.9, do_sample=True,pad_token_id=self.llm_tokenizer.eos_token_id)
                response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                response = self._get_fallback_response(user_message, var_context)
            return response if response else self._get_fallback_response(user_message, var_context)
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return self._get_fallback_response(user_message, var_context)
    def _get_fallback_response(self, user_message: str, var_context: str) -> str:
        print("enforcing fallback response due to LLM error or unavailability.")
        response = "I'm here to help with your VaR predictions and market insights.\n\n"

        if "var" in user_message.lower() or "risk" in user_message.lower():
            response += f"Based on the current VaR context: {var_context}\n"
            response += "Please let me know if you have specific questions about the VaR calculations or need insights on how to interpret the results.\n"
        else:
            response += "Please ask me about:\n"
            response += "- Understanding VaR predictions\n"
            response += "- Interpreting market risks\n"
            response += "- Insights based on current market news\n"
        return response

@st.cache_resource
def get_news_agent() -> NewsEmbeddingAgent:
    return NewsEmbeddingAgent()
