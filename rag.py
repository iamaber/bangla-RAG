import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Database and ML Libraries
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pymysql

# ML and NLP Libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

class MySQLDatabaseConfig:
    def __init__(self, host, user, password, database):
        self.connection_string = f'mysql+pymysql://root:abir1234@localhost/news_db'
        
        # Create SQLAlchemy engine with UTF-8 support
        self.engine = create_engine(
            self.connection_string,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False
        )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)

class BanglaNewsRAG:
    def __init__(self, db_config: MySQLDatabaseConfig):
        self.db_config = db_config
        
        # Initialize Multilingual Embedding Model
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize Keyword Extraction
        self.keyword_extractor = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from article text
        
        Args:
            text (str): Article body text
            top_n (int): Number of keywords to extract
        
        Returns:
            List[str]: Extracted keywords
        """
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',  # You might want a Bangla stop words list
                top_n=top_n
            )
            return [keyword for keyword, _ in keywords]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []
    
    def semantic_search(self, query: str, articles: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search on articles
        
        Args:
            query (str): Search query
            articles (List[Dict]): List of article dictionaries
            top_k (int): Number of top results to return
        
        Returns:
            List[Dict]: Top relevant articles
        """
        # Extract article bodies
        article_texts = [article['article_body'] for article in articles]
        
        # Get embeddings
        query_embedding = self.embedding_model.encode([query])[0]
        article_embeddings = self.embedding_model.encode(article_texts)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], article_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [articles[i] for i in top_indices]
    
    def extract_statistics(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Extract numerical statistics from articles
        
        Args:
            articles (List[Dict]): List of articles
        
        Returns:
            Dict[str, Any]: Statistical summary
        """
        def extract_numbers(text: str) -> List[int]:
            # Extract numbers from text (works with both English and Bangla numerals)
            number_pattern = re.compile(r'\d+')
            return [int(num) for num in number_pattern.findall(text)]
        
        all_numbers = []
        for article in articles:
            all_numbers.extend(extract_numbers(article['article_body']))
        
        if not all_numbers:
            return None
        
        return {
            'total': sum(all_numbers),
            'average': np.mean(all_numbers),
            'max': max(all_numbers),
            'min': min(all_numbers),
            'count': len(all_numbers)
        }
    
    def generate_response(self, query: str, articles: List[Dict], statistics: Dict[str, Any] = None) -> str:
        """
        Generate a response in Bangla
        
        Args:
            query (str): Original query
            articles (List[Dict]): Relevant articles
            statistics (Dict[str, Any]): Statistical summary
        
        Returns:
            str: Formatted response
        """
        # Response parts
        response_parts = []
        
        # Add statistical summary if available
        if statistics:
            stats_text = (
                f"📊 পরিসংখ্যান বিশ্লেষণ:\n"
                f"- মোট সংখ্যা: {statistics['total']}\n"
                f"- গড় মান: {statistics['average']:.2f}\n"
                f"- সর্বোচ্চ মান: {statistics['max']}\n"
                f"- সর্বনিম্ন মান: {statistics['min']}"
            )
            response_parts.append(stats_text)
        
        # Add sources
        sources = "\n\n🔗 সংবাদ সূত্রসমূহ:"
        for idx, article in enumerate(articles, 1):
            sources += f"\n{idx}. {article['article_link']}"
        
        response_parts.append(sources)
        
        return "\n".join(response_parts)
    
    def process_query(self, query: str) -> str:
    
        with self.db_config.Session() as session:
        # Retrieve recent articles (last 30 days)
            articles_query = text("""
            SELECT 
                id, 
                headline, 
                article_body, 
                article_link, 
                date_published 
            FROM articles 
            WHERE date_published >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """)
        
        result = session.execute(articles_query)
        
        # Properly convert result to list of dictionaries
        articles = [
            {
                'id': row.id,
                'headline': row.headline,
                'article_body': row.article_body,
                'article_link': row.article_link,
                'date_published': row.date_published
            } 
            for row in result
        ]
    
    # If no articles found, return a message
        if not articles:
            return "দুঃখিত, কোনো সাম্প্রতিক নিবন্ধ পাওয়া যায়নি।"
    
    # Perform semantic search
        relevant_articles = self.semantic_search(query, articles)
    
    # Extract statistics
        statistics = self.extract_statistics(relevant_articles)
    
    # Generate response
        response = self.generate_response(query, relevant_articles, statistics)
    
        return response

import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        # Database Configuration
        db_config = MySQLDatabaseConfig(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )
        
        # Initialize RAG System
        rag_system = BanglaNewsRAG(db_config)
        
        # Example Queries
        queries = [
            "গত মাসে কতগুলি সড়ক দুর্ঘটনা ঘটেছে?",
            "করোনা ভাইরাসের সর্বশেষ তথ্য কী?",
            "রাজনৈতিক পরিস্থিতি কী রকম?"
        ]
        
        # Process and print responses
        for query in queries:
            logging.info(f"Processing query: {query}")
            try:
                response = rag_system.process_query(query)
                print(f"\n🔍 Query: {query}")
                print(response)
                logging.info(f"Response generated successfully")
            except Exception as query_error:
                logging.error(f"Error processing query '{query}': {query_error}")
                logging.error(traceback.format_exc())
    
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
        logging.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
