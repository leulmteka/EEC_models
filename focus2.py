#!/usr/bin/env python3
"""
Churro Chiller Interviewer with OpenRouter GPT-4, Dynamic Focused Follow-ups, and RAG Integration
-----------------------------------------------------------------------------------------------
This script conducts an in-depth, dynamically focused interview about Taco Bell’s new “Churro Chiller.”
It uses a Retrieval Augmented Generation (RAG) component (via TF-IDF) to supply relevant context
and an interview guide, and uses the OpenRouter API (accessed via HTTP POST with the requests library)
to generate on-topic follow-up questions. The interviewer remains focused on the current topic,
iteratively asking follow-ups that probe for specific details until the interviewee signals no further
detail is available. Once enough detail is gathered for a topic, the interview moves on.

Requirements:
  - Install scikit-learn, requests, and python-dotenv:
      pip install scikit-learn requests python-dotenv

Usage:
  - Set your OpenRouter API key in the environment variable OPENROUTER_API_KEY. For example:
        export OPENROUTER_API_KEY='your_openrouter_api_key'
"""

import os
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from a .env file if available.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Define headers for our OpenRouter API requests.
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

class RAGRetriever:
    """
    RAG component using TF-IDF to index and retrieve context documents.
    
    Documents Indexed:
      - "Churro Chiller": Contains detailed product information.
      - "Interview Guide": Contains the interview framework and prompt ideas.
    """
    def __init__(self):
        self.documents = {
            "Churro Chiller": (
                "The Churro Chiller is a new beverage by Taco Bell, similar to a Baja Blast but with a cinnamon twist. "
                "It retains the caffeine and fizz of a frozen Mountain Dew-based drink, contains no artificial colorings or additives, "
                "and is offered at a discount when purchased with a Cheesy Gordita Crunch or a Crunchwrap Supreme."
            ),
            "Interview Guide": (
                "===================================================\n"
                "Churro Chiller Interview Guide for RAG Implementation\n"
                "===================================================\n\n"
                "Purpose:\n"
                "----------\n"
                "This guide is designed to stimulate a natural, open-ended conversation about Taco Bell's new “Churro Chiller.” "
                "It covers key topics—overall interest, flavor expectations, experience/enjoyment, pairing with food, price sensitivity, "
                "and comparative feedback—while offering follow-up ideas to elicit deeper insights.\n\n"
                "---------------------------------------------------\n"
                "Section 1: Introduction & Overall Impressions\n"
                "---------------------------------------------------\n"
                "Primary Question:\n"
                "   - What are your overall thoughts about a Churro Chiller—a cinnamon-infused, caffeinated, fizzy frozen drink?\n\n"
                "---------------------------------------------------\n"
                "Section 2: Flavor Profile & Sensory Expectations\n"
                "---------------------------------------------------\n"
                "Primary Question:\n"
                "   - How would you describe the flavor profile you expect from a cinnamon-flavored, fizzy drink?\n\n"
                "---------------------------------------------------\n"
                "Section 3: Experience & Enjoyment\n"
                "---------------------------------------------------\n"
                "Primary Question:\n"
                "   - Have you had a chance to try the Churro Chiller? If not, what do you imagine it might be like?\n\n"
                "---------------------------------------------------\n"
                "Section 4: Menu Pairings & Usage\n"
                "---------------------------------------------------\n"
                "Primary Question:\n"
                "   - Taco Bell offers a discount when the Churro Chiller is paired with items like a Cheesy Gordita Crunch or "
                "a Crunchwrap Supreme. What do you think about that pairing?\n\n"
                "---------------------------------------------------\n"
                "Section 5: Comparisons & Price\n"
                "---------------------------------------------------\n"
                "Primary Questions:\n"
                "   - How does the Churro Chiller compare to other beverages available?\n"
                "   - What would you consider a reasonable price for it?\n\n"
                "---------------------------------------------------\n"
                "Usage Instructions:\n"
                "   - Use the primary questions to initiate discussion on each topic.\n"
                "   - Dynamically generate follow-up questions based on user responses, bouncing back on new details, "
                "asking for clarifications, examples, and elaborations.\n"
                "   - Remain strictly focused on the current topic to gain in-depth feedback before moving on.\n"
                "===================================================\n"
            )
        }
        self.doc_ids = list(self.documents.keys())
        self.corpus = list(self.documents.values())
        self.vectorizer = TfidfVectorizer().fit(self.corpus)
        self.doc_vectors = self.vectorizer.transform(self.corpus)

    def get_context(self, query: str, top_n: int = 2) -> str:
        """
        Retrieve the top_n most relevant documents based on cosine similarity.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        ranked_idx = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_idx[:top_n]:
            if similarities[idx] > 0:
                doc_id = self.doc_ids[idx]
                results.append(f"{doc_id}:\n{self.documents[doc_id]}")
        return "\n\n".join(results) if results else "No relevant context found."

class ChurroChillerInterviewer:
    """
    GPT-based Interviewer that conducts an in-depth, dynamically focused interview on the Churro Chiller
    using OpenRouter’s API (via HTTP POST with requests) and RAG.
    
    For each topic:
      - Ask the main question.
      - Iteratively generate focused follow-up questions using conversation history and the current aggregated response.
      - Aggregate all details until the interviewee signals no further detail, then move to the next topic.
    """
    def __init__(self):
        self.retriever = RAGRetriever()
        # Each topic's response is stored as an aggregated string.
        self.topics = {
            "interest": "",
            "flavor_impression": "",
            "enjoyment": "",
            "menu_pairings": "",
            "comparisons": "",
            "willingness_to_pay": ""
        }
        self.questions = [
            "What are your overall thoughts about a Churro Chiller—a cinnamon-infused, caffeinated, fizzy frozen drink?",
            "How would you describe the flavor profile you expect from a cinnamon-flavored, fizzy drink?",
            "Have you had a chance to try the Churro Chiller? If not, what do you imagine it might be like?",
            "Taco Bell offers a discount when the Churro Chiller is paired with items like a Cheesy Gordita Crunch or a Crunchwrap Supreme. What do you think about that pairing?",
            "How does the Churro Chiller compare to other beverages available?",
            "What would you consider a reasonable price for a beverage like the Churro Chiller?"
        ]
        self.topic_keys = list(self.topics.keys())
        self.current_topic_index = 0
        self.history = ""  # Cumulative conversation history.
        self.max_followups = 3  # Maximum follow-up iterations per topic.

    def retrieve_context(self):
        """
        Retrieve context from product information and the interview guide.
        """
        product_info = self.retriever.get_context("Churro Chiller")
        interview_guide = self.retriever.get_context("Interview Guide")
        return {"product_info": product_info, "interview_guide": interview_guide}

    def generate_followup(self, topic: str, aggregated_response: str) -> str:
        """
        Generate a focused follow-up question using OpenRouter's chat completions API.
        The question will remain strictly on the current topic and probe deeper into the details provided.
        """
        prompt = (
            "You are a highly focused and inquisitive interviewer. Follow these guidelines:\n"
            "  - Your follow-up question must be strictly about the current topic: {}\n"
            "  - Ask open-ended questions that require more than a yes/no answer.\n"
            "  - Bounce back on specific details from the aggregated response by asking for examples, clarifications, or further elaboration.\n"
            "  - Remain strictly focused on the topic; do not deviate to unrelated issues.\n\n"
            "Conversation History:\n{}\n\n"
            "Current aggregated response for '{}': '{}'\n\n"
            "Based on the above, generate a specific follow-up question to gain more in-depth insight."
        ).format(topic, self.history, topic, aggregated_response)
        
        data = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [
                {"role": "system", "content": "You are a curious user interviewer who wants to know about the current topic."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(API_URL, json=data, headers=HEADERS)
            response.raise_for_status()
            resp_json = response.json()
            followup = resp_json["choices"][0]["message"]["content"].strip()
            if not followup.lower().startswith(("what", "how", "why", "could", "tell me")):
                followup = "Could you please elaborate further on that?"
            return followup
        except Exception as e:
            print("Error generating follow-up:", e)
            return "Could you please elaborate further on that?"

    def ask_topic(self, topic_idx: int):
        """
        Ask the main question for the current topic and iteratively ask focused follow-up
        questions until the interviewee signals no further detail or until the maximum number of follow-ups is reached.
        """
        topic_key = self.topic_keys[topic_idx]
        primary_question = self.questions[topic_idx]
        print("\nInterviewer: " + primary_question)
        self.history += f"Interviewer: {primary_question}\n"
        aggregated_response = input("You: ").strip()
        self.history += f"User: {aggregated_response}\n"
        
        followup_count = 0
        while True:
            followup = self.generate_followup(topic_key, aggregated_response)
            if not followup:
                break
            print("Interviewer: " + followup)
            self.history += f"Interviewer (Follow-up on {topic_key}): {followup}\n"
            followup_response = input("You: ").strip()
            self.history += f"User: {followup_response}\n"
            if followup_response.lower() in ["no", "nothing", "no more", "that's all", "done"]:
                break
            aggregated_response += " " + followup_response
            followup_count += 1
            if followup_count >= self.max_followups:
                break
        
        self.topics[topic_key] = aggregated_response
        print(f"\n--- End of topic '{topic_key}' ---\n")

    def run_interview(self):
        """
        Conduct the full interview by iterating through each topic.
        Conclude with a final open-ended wrap-up question.
        """
        contexts = self.retrieve_context()
        print("RAG Context - Product Information:")
        print(contexts["product_info"])
        print("\nRAG Context - Interview Guide:")
        print(contexts["interview_guide"])
        print("-" * 80)
        
        while self.current_topic_index < len(self.topic_keys):
            self.ask_topic(self.current_topic_index)
            self.current_topic_index += 1

        wrapup_question = "Is there anything else you would like to add or any other feedback on the Churro Chiller?"
        print("Interviewer: " + wrapup_question)
        self.history += f"Interviewer (Wrap-up): {wrapup_question}\n"
        final_comments = input("You: ").strip()
        self.history += f"User: {final_comments}\n"

        print("\nInterviewer: Thank you for your detailed feedback on the Churro Chiller. Your insights are very valuable!")
        print("\n--- Aggregated Interview Responses ---")
        for topic, response in self.topics.items():
            print(f"\n{topic.capitalize()}:")
            print(response)
        if final_comments:
            print("\nAdditional Comments:")
            print(final_comments)

def main():
    """
    Entry point for running the dynamic, in-depth Churro Chiller Interviewer with OpenRouter GPT-4.
    """
    interviewer = ChurroChillerInterviewer()
    interviewer.run_interview()

if __name__ == "__main__":
    main()
