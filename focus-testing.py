#!/usr/bin/env python3
"""
deepseek_interview.py

A conversational interview conductor for Taco Bell's new product—the Churro Chiller.
Rather than directly asking for each piece of target information, the system gently
leads the human user into a discussion that reveals:
  - Overall interest in the product.
  - Impressions of its cinnamon-infused flavor.
  - Enjoyment levels.
  - Price sensitivity.
  - Best pairing suggestions with Taco Bell menu items.
  - Preference relative to other beverage choices.

Product Details:
  - The Churro Chiller is like a Baja Blast but with a cinnamon twist.
  - It has the caffeine and fizz of a Mountain Dew based frozen drink.
  - No artificial colorings or additives are used.
  - It is sold at a discount when paired with a Cheesy Gordita Crunch or a Crunchwrap Supreme.

The interview gently follows up on the user’s input to gather details without asking overly direct questions.
"""

import argparse
import re

# ------------------------------------------------------------------------------
# RAG Component: Assembles static product details with conversation context.
# ------------------------------------------------------------------------------

class RAGClient:
    def __init__(self, product_details: str):
        self.product_details = product_details

    def retrieve_context(self, conversation_history: dict) -> str:
        """
        Returns a summary of the static product details and the key points gathered so far.
        
        Args:
            conversation_history (dict): Dictionary of topic keys to aggregated responses.
        
        Returns:
            str: A text summary of context.
        """
        context_parts = [f"Product Details: {self.product_details}"]
        for topic, response in conversation_history.items():
            if response:
                context_parts.append(f"{topic.capitalize()} info: {response}")
        return "\n".join(context_parts)


# ------------------------------------------------------------------------------
# LLM Component: Decides what follow-up question to ask based on conversation state.
# ------------------------------------------------------------------------------

class LLMClient:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

        # For each topic, we use a set of keywords to detect if the topic has been touched on.
        self.topic_keywords = {
            "interest": ["interested", "excited", "curious", "appeal", "like", "love", "intrigued"],
            "flavor": ["cinnamon", "flavor", "taste", "spicy", "sweet", "bitter", "churro"],
            "enjoyment": ["enjoy", "good", "fun", "delicious", "tasty", "dislike", "bad"],
            "price": ["price", "pay", "cost", "expensive", "cheap", "value", "afford"],
            "pairing": ["pair", "with", "combo", "meal", "gordita", "crunchwrap", "taco", "food"],
            "preference": ["better", "prefer", "choice", "other", "alternative", "compare"]
        }

        # Follow-up prompts are designed to prompt more discussion on a topic without direct questioning.
        self.followup_prompts = {
            "interest": (
                "You mentioned your feelings about trying new things. Could you share a bit more about what excites or concerns you about this kind of product?"
            ),
            "flavor": (
                "I'd love to hear more about what you think regarding its taste. What sort of flavor nuances come to your mind when you picture a cinnamon twist in a familiar frozen drink?"
            ),
            "enjoyment": (
                "It sounds like you have some impressions about enjoying such a drink. Can you describe what aspects of the experience are most appealing or not appealing to you?"
            ),
            "price": (
                "Pricing can make a big difference. How do you feel about the value of a drink like this, especially if offered at a discount with certain menu items?"
            ),
            "pairing": (
                "Pairing drinks with food can elevate the experience. What combinations with Taco Bell’s items do you think would really work with this drink?"
            ),
            "preference": (
                "Sometimes people have clear preferences among available drink options. How does this product measure up to other beverages you enjoy at Taco Bell?"
            )
        }
    
    def analyze_response_for_topics(self, response: str) -> dict:
        """
        Look for keywords in the user's response to update coverage of each topic.
        
        Args:
            response (str): The user's input.
            
        Returns:
            dict: A mapping of topic -> True if any keywords were detected.
        """
        findings = {topic: False for topic in self.topic_keywords}
        lower_response = response.lower()
        for topic, keywords in self.topic_keywords.items():
            for kw in keywords:
                # Use word boundaries to avoid partial matches.
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_response):
                    findings[topic] = True
                    break
        return findings

    def select_followup(self, conversation_history: dict, response: str) -> str:
        """
        Decide on the best follow-up prompt based on what target topics are touched upon or missing.
        
        Args:
            conversation_history (dict): Aggregated responses per topic.
            response (str): The latest user input.
            
        Returns:
            str: A follow-up prompt designed to encourage more detailed discussion.
        """
        # Analyze the current response.
        detected = self.analyze_response_for_topics(response)
        
        # For any topic that is still missing or needs more detail, and that was detected in the latest response,
        # offer a follow-up question. If none of the topics has been mentioned yet, ask a general prompt.
        for topic, is_detected in detected.items():
            if is_detected and conversation_history.get(topic) is None:
                # First time mention: store the response for that topic.
                conversation_history[topic] = response
                return self.followup_prompts[topic]
            # If topic already recorded but the conversation remains sparse, we can still probe:
            elif is_detected and conversation_history.get(topic):
                # Optionally, check if the stored response is very short (less than, say, 5 words) to request more detail.
                if len(conversation_history[topic].split()) < 5:
                    return self.followup_prompts[topic]
        
        # If no specific target topic was identified in this turn, or all mentioned topics are already probed,
        # offer a more general invitation for additional thoughts.
        return "Can you tell me more about your thoughts on this new drink? Feel free to share any details that come to mind."

    def generate_initial_prompt(self) -> str:
        """
        Generate an initial, open-ended prompt to start the conversation in a natural way.
        
        Returns:
            str: The initial prompt.
        """
        return ("Taco Bell is considering launching a new beverage called the Churro Chiller—a frozen drink with a "
                "cinnamon twist that delivers the caffeine and fizz of a Mountain Dew based frozen drink. What do you think about "
                "the idea of trying something like that?")

# ------------------------------------------------------------------------------
# Interview Conductor: Manages the conversation and stateful probing.
# ------------------------------------------------------------------------------

class DeepSeekInterview:
    def __init__(self, product: str, system_prompt: str):
        """
        Initialize interview state and load the product details.
        
        Args:
            product (str): The product name (expected to be 'Churro Chiller').
            system_prompt (str): A system-level instruction (can be used to set tone).
        """
        self.product = product
        self.system_prompt = system_prompt
        self.product_details = (
            "The Churro Chiller is like a Baja Blast but with a cinnamon twist. It retains the caffeine and fizz of a "
            "Mountain Dew based frozen drink, contains no artificial colorings or additives, and is sold at a discount "
            "when purchased with a Cheesy Gordita Crunch or a Crunchwrap Supreme."
        )
        # We'll track the conversation as a mapping from target topic to the aggregated response.
        self.conversation_history = {
            "interest": None,
            "flavor": None,
            "enjoyment": None,
            "price": None,
            "pairing": None,
            "preference": None
        }
        self.rag_client = RAGClient(self.product_details)
        self.llm_client = LLMClient(self.system_prompt)
        self.overall_conversation = []  # store full conversation logs (role, message)

    def record_message(self, role: str, message: str):
        """Keep a full log of the conversation."""
        self.overall_conversation.append((role, message))

    def review_overall_context(self) -> str:
        """
        Combine static product details with the conversation history for context.
        
        Returns:
            str: The conversation context.
        """
        return self.rag_client.retrieve_context(self.conversation_history)

    def update_topics(self, response: str):
        """
        Update conversation history with detected topics based on the new response.
        
        Args:
            response (str): The user's latest input.
        """
        detected = self.llm_client.analyze_response_for_topics(response)
        for topic, present in detected.items():
            if present:
                # If this is the first time a topic is mentioned, record it.
                if self.conversation_history[topic] is None:
                    self.conversation_history[topic] = response
                else:
                    # Append to the existing discussion for the topic.
                    self.conversation_history[topic] += " " + response

    def all_topics_addressed(self) -> bool:
        """
        Check if all target topics have been addressed in some form.
        
        Returns:
            bool: True if every topic has received a response, else False.
        """
        return all(response is not None for response in self.conversation_history.values())

    def run_interview(self):
        """
        Conduct the conversational interview until all target topics receive sufficient discussion.
        """
        print(f"Welcome! Today we'll chat about Taco Bell's upcoming product: {self.product}.")
        print("Feel free to share your thoughts—there are no right or wrong answers. Type 'exit' to quit at any time.\n")

        # Start with an initial open-ended prompt.
        initial_prompt = self.llm_client.generate_initial_prompt()
        print(f"Interview Prompt: {initial_prompt}\n")
        self.record_message("System", initial_prompt)

        while True:
            try:
                user_input = input("You: ").strip()
            except KeyboardInterrupt:
                print("\nInterview terminated by user.")
                break

            if user_input.lower() in ['exit', 'quit']:
                print("Interview ended by user. Thank you for your time!")
                break

            self.record_message("User", user_input)
            self.update_topics(user_input)

            # Generate an appropriate follow-up based on the latest response and overall context.
            followup = self.llm_client.select_followup(self.conversation_history, user_input)
            print(f"\nFollow-up: {followup}\n")
            self.record_message("System", followup)

            # Optionally, show current context for debugging purposes (comment out if undesired).
            # print("----- Conversation Context -----")
            # print(self.review_overall_context())
            # print("--------------------------------")

            # End the interview if all topics seem addressed.
            if self.all_topics_addressed():
                print("It looks like we've covered many aspects of the Churro Chiller! Thank you for the insightful discussion.")
                print("\nSummary of key points:")
                summary_lines = [
                    f"Interest: {self.conversation_history['interest']}",
                    f"Flavor Impression: {self.conversation_history['flavor']}",
                    f"Enjoyment Thoughts: {self.conversation_history['enjoyment']}",
                    f"Price and Value: {self.conversation_history['price']}",
                    f"Food Pairing: {self.conversation_history['pairing']}",
                    f"Preference over Other Drinks: {self.conversation_history['preference']}"
                ]
                print("\n".join(summary_lines))
                break

# ------------------------------------------------------------------------------
# Main entry point: Parse arguments and start the interview session.
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A conversational interview conductor for Taco Bell's Churro Chiller using RAG principles."
    )
    parser.add_argument("--product", default="Churro Chiller", help="Product to be discussed (default: Churro Chiller)")
    parser.add_argument("--system_prompt", required=True, help="System prompt for guiding the interview tone and style")
    args = parser.parse_args()

    interview = DeepSeekInterview(args.product, args.system_prompt)
    interview.run_interview()

if __name__ == '__main__':
    main()
