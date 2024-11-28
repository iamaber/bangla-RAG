import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict

class NewsRAG:
    def __init__(self):
        # Previous initialization code remains the same
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="abir1234",
            database="news_db"
        )
        self.cursor = self.db.cursor(dictionary=True)

    def get_embeddings(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)
    
    def search_articles(self, query: str, top_k: int = 5) -> List[Dict]:
        # Get query embedding
        query_embedding = self.get_embeddings(query)
        
        # Get relevant articles from the last 3 months
        three_months_ago = datetime.now() - timedelta(days=90)
        sql = """
        SELECT id, date_published, headline, article_body, article_link, article_site 
        FROM articles 
        WHERE date_published >= %s
        """
        self.cursor.execute(sql, (three_months_ago,))
        articles = self.cursor.fetchall()
        
        # Calculate similarities
        similarities = []
        for article in articles:
            article_text = f"{article['headline']} {article['article_body']}"
            article_embedding = self.get_embeddings(article_text)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                article_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, article))
        
        # Sort by similarity and get top_k results
        similarities.sort(reverse=True)
        return [article for _, article in similarities[:top_k]]
    
    def extract_numbers_from_text(self, text: str) -> List[Tuple[str, str]]:
        # Bengali numbers and their Arabic numeral equivalents
        bengali_numbers = "০১২৩৪৫৬৭৮৯"
        arabic_numbers = "0123456789"
        trans_table = str.maketrans(bengali_numbers, arabic_numbers)
        
        # Convert Bengali numbers to Arabic numerals
        text = text.translate(trans_table)
        
        # Find numbers with context
        # Looking for numbers and the surrounding words
        number_contexts = []
        
        # Pattern to match numbers (including decimals) and grab surrounding context
        pattern = r'([^।]*?\d+(?:\.\d+)?[^।]*?।)'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            context = match.group(1).strip()
            number_contexts.append(context)
            
        return number_contexts

    def get_statistics(self, query: str) -> Tuple[str, List[Dict]]:
        # Get relevant articles
        relevant_articles = self.search_articles(query, top_k=5)
        
        if not relevant_articles:
            return "দুঃখিত, আপনার প্রশ্নের সাথে সম্পর্কিত কোন তথ্য পাওয়া যায়নি।", []
        
        # Extract statistical information from articles
        statistical_info = []
        
        for article in relevant_articles:
            # Combine headline and body for searching
            full_text = f"{article['headline']} {article['article_body']}"
            
            # Extract sentences containing numbers
            number_contexts = self.extract_numbers_from_text(full_text)
            
            if number_contexts:
                article['statistical_contexts'] = number_contexts
                statistical_info.append(article)
        
        # Format response
        if statistical_info:
            response = f"আপনার প্রশ্নের সংক্রান্ত পরিসংখ্যানগত তথ্য:\n\n"
            
            for article in statistical_info:
                response += f"📅 {article['date_published'].strftime('%d-%m-%Y')}:\n"
                for context in article['statistical_contexts'][:3]:  # Limit to top 3 statistical contexts per article
                    response += f"• {context}\n"
                response += "\n"
        else:
            response = "প্রাসঙ্গিক লেখাগুলিতে কোন পরিসংখ্যানগত তথ্য পাওয়া যায়নি।"
        
        return response, statistical_info

    def format_response_with_sources(self, response: str, articles: List[Dict]) -> str:
        formatted_response = f"{response}\nতথ্যসূত্র:"
        for i, article in enumerate(articles, 1):
            formatted_response += f"\n{i}. {article['article_site']} ({article['date_published'].strftime('%d-%m-%Y')})"
            formatted_response += f"\n   লিंক: {article['article_link']}"
        return formatted_response

def main():
    rag = NewsRAG()
    
    while True:
        # query = input("\nআপনার প্রশ্ন লিখুন (প্রস্থান করতে 'exit' লিখুন): ")
        query = "গত মাসে কতগুলি সড়ক দুর্ঘটনা ঘটেছে?"
        if query.lower() == 'exit':
            break
            
        response, articles = rag.get_statistics(query)
        formatted_response = rag.format_response_with_sources(response, articles)
        print("\n" + formatted_response)

if __name__ == "__main__":
    main()