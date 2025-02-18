from typing import Dict, List

import numpy as np
from autogen import ConversableAgent, register_function
import sys
import os


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # TODO
    # This function takes in a restaurant name and returns the reviews for that restaurant.
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call.
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    true_name = None
    with open("restaurant-data.txt") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.lower()

            def f(temp):
                temp = temp.lower()
                temp = temp.replace('"', "")
                temp = temp.replace(",", " ")
                temp = temp.replace(".", " ")
                temp = temp.replace("-", " ")
                return temp

            temp = f(temp)
            simple_restaurant_name = f(restaurant_name)

            if temp.startswith(simple_restaurant_name):
                if true_name is None:
                    true_name = restaurant_name
                    ans = {true_name: list()}
                ans[true_name].append(line)
    return ans


def calculate_overall_score(
    restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]
) -> Dict[str, float]:
    food_scores = np.array(food_scores)
    customer_service_scores = np.array(customer_service_scores)
    ans = (
        np.sum(
            np.sqrt(food_scores**2 * customer_service_scores)
            / len(food_scores)
            / np.sqrt(125)
        )
        * 10
    )
    if int(ans) == ans:
        ans += 0.000001
    return {restaurant_name: ans}


def main(user_query: str):
    entrypoint_agent_system_message = """You are the supervisor agent coordinating the restaurant review analysis process.
        Follow these steps exactly:
        1. First, ask the data fetch agent to get restaurant reviews using fetch_restaurant_data
        2. Once you have the reviews, send them to the review analyzer to extract scores
        3. After getting the scores from the analyzer, ask the scoring agent to calculate the final rating
    """
    scores_mapping = f"""Score 1/5 has one of these adjectives: awful, horrible, or disgusting.
    Score 2/5 has one of these adjectives: bad, unpleasant, or offensive.
    Score 3/5 has one of these adjectives: average, uninspiring, or forgettable.
    Score 4/5 has one of these adjectives: good, enjoyable, or satisfying.
    Score 5/5 has one of these adjectives: awesome, incredible, or amazing."""
    fetch_agent_prompt = f"""You are a data fetch agent responsible for extracting restaurant names from user queries 
    and fetching their reviews.

    Your task:
    1. Analyze the user query: "{user_query}"
    2. Extract the restaurant name from the query
    3. Call the fetch_restaurant_data function with the extracted name
    """
    analysis_agent_prompt = f"""You are a review analyzer agent. 
    Your task is to analyze restaurant reviews and extract scores.
    
    For each review:
    1. Find exactly one keyword for food quality and one for service quality
    2. Map keywords to scores using this exact mapping:
        Food/Service Score Mapping:
    
    {scores_mapping}
    
    Output format must be exactly:
    food_scores = [score1, score2, ...]
    customer_service_scores = [score1, score2, ...]
    """
    scorer_agent_prompt = f"""You are a scoring agent. Your task is to take the food scores and customer service scores from the previous conversation and calculate the final rating.

    Steps:
    1. Extract the restaurant name from the data fetch result
    2. Get the food_scores and customer_service_scores lists from the analyzer
    3. Call calculate_overall_score with these exact parameters
    """
    llm_config = {
        "config_list": [
            {
                "model": "qvikhr",
                "base_url": "http://localhost:7777/v1",
                "api_key": "api",
            }
        ]
    }

    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config,
    )

    fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=fetch_agent_prompt,
        llm_config=llm_config,
    )

    analyzer = ConversableAgent(
        "review_analyzer_agent",
        system_message=analysis_agent_prompt,
        llm_config=llm_config,
    )

    scorer_agent = ConversableAgent(
        "scoring_agent",
        system_message=scorer_agent_prompt,
        llm_config=llm_config,
    )

    register_function(
        fetch_restaurant_data,
        caller=entrypoint_agent,
        executor=fetch_agent,
        name="fetch_restaurant_data",
        description="Fetches the reviews for specific restaurant"
    )

    register_function(
        calculate_overall_score,
        caller=entrypoint_agent,
        executor=scorer_agent,
        name="calculate_overall_score",
        description="Calculates the overall score for a restaurant",
    )

    chat_results = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": fetch_agent,
                "message": f"Find reviews for this query: {user_query}",
                "summary_method": "last_msg",
                "max_turns": 2,
            },
            {
                "recipient": analyzer,
                "message": f"Here are the reviews from the fetch agent. "
                           f"Please analyze them and extract food and service scores. "
                           f"For each review, find the food quality keyword and service quality keyword, "
                           f"then map them to scores 1-5 according to the scoring rules.",
                "summary_method": "last_msg",
                "max_turns": 1,
            },
            {
                "recipient": scorer_agent,
                "message": f"Here are the reviews from the analysis agent, "
                           f"please calculate the final restaurant rating using calculate_overall_score ",
                "summary_method": "last_msg",
                "max_turns": 2,
            }
        ]
    )
    print(chat_results)
    return chat_results


# DO NOT modify this code below.
if __name__ == "__main__":
    assert (
        len(sys.argv) > 1
    ), "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
