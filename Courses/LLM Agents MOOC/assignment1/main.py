from typing import Dict, List

import numpy as np
from autogen import ConversableAgent, AssistantAgent
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
                temp = temp.replace("\"", "")
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


def get_scores(
    restaurant_name: str, restaurant_data: Dict[str, List[str]]
) -> tuple[List[int], List[int]]:
    """
    Score 1/5 has one of these adjectives: awful, horrible, or disgusting.
    Score 2/5 has one of these adjectives: bad, unpleasant, or offensive.
    Score 3/5 has one of these adjectives: average, uninspiring, or forgettable.
    Score 4/5 has one of these adjectives: good, enjoyable, or satisfying.
    Score 5/5 has one of these adjectives: awesome, incredible, or amazing.
    :param restaurant_name:
    :param restaurant_data:
    :return:
    """
    values = restaurant_data.get(restaurant_name, None)
    food_array = list()
    service_array = list()
    scores = {
        "awful": 1,
        "horrible": 1,
        "disgusting": 1,
        "bad": 2,
        "unpleasant": 2,
        "offensive": 2,
        "average": 3,
        "uninspiring": 3,
        "forgettable": 3,
        "good": 4,
        "joy": 4,
        "satisfying": 4,
        "awesome": 5,
        "incredibl": 5,
        "amaz": 5,
    }
    for value in values:
        service_index = max(
            value.find("staff"), value.find("service"), value.find("baristas")
        )
        indices = list()
        marks = list()
        for key in scores.keys():
            if value.find(key) >= 0:
                indices.append(value.find(key))
                marks.append(scores[key])
        #assert len(indices) == len(marks) == 2
        if indices[0] > indices[1]:
            indices[0], indices[1] = indices[1], indices[0]
            marks[0], marks[1] = marks[1], marks[0]
        if service_index < indices[1]:
            service_array.append(marks[0])
            food_array.append(marks[1])
        else:
            service_array.append(marks[1])
            food_array.append(marks[0])
    return food_array, service_array


def calculate_overall_score(
    restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]
) -> Dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service.
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.048}
    # NOTE: be sure to that the score includes AT LEAST 3  decimal places. The public tests will only read scores that have
    # at least 3 decimal places.
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


def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    # TODO
    # It may help to organize messages/prompts within a function which returns a string.
    # For example, you could use this function to return a prompt for the data fetch agent
    # to use to fetch reviews for a specific restaurant.
    pass


# TODO: feel free to write as many additional functions as you'd like.


# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = (
        "Manage the scoring pipeline: \n"
        "1. Extract restaurant name using NameExtractor \n"
        "2. Fetch data using fetch_restaurant_data\n"
        "3. Analyze with analyze_sentiment\n"
        "4. Calculate final score with calculate_overall\n"
        "Present final score with explanation"
    )  # TODO
    # example LLM config for the entrypoint agent
    # llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    llm_config = {
        "config_list": [
            {
                "model": "qvikhr",
                "base_url": "http://localhost:7777/v1",  # Replace with your local server address
                "api_key": "api",  # Many local setups don't require an API key
            }
        ]
    }
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
    )
    name_extractor = AssistantAgent(
        name="NameExtractor",
        system_message="Extract ONLY the restaurant name from user queries. Respond ONLY with the name.",
        llm_config=llm_config,
    )
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant.",
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(
        fetch_restaurant_data
    )

    # TODO
    # Create more agents here.
    entrypoint_agent.register_for_llm(
        name="get_scores", description="Get scores from text provided"
    )(get_scores)
    entrypoint_agent.register_for_execution(name="get_scores")(get_scores)
    entrypoint_agent.register_for_llm(
        name="calculate_overal_score",
        description="Calculate overal score for a restaurant",
    )(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(
        calculate_overall_score
    )
    chat_results = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": name_extractor,
                "message": user_query,
                "clear_history": True,
                "max_turns": 2,
            }
        ]
    )
    # Assume the last message in the chat history is the extracted restaurant name.
    restaurant_name = chat_results[0].chat_history[-1]["content"].strip()

    # Step 2: Fetch restaurant data (reviews) using the provided function.
    restaurant_data = fetch_restaurant_data(restaurant_name)

    # Step 3: Get food and customer service scores from the reviews.
    food_scores, service_scores = get_scores(restaurant_name, restaurant_data)

    # Step 4: Calculate the overall score.
    overall = calculate_overall_score(restaurant_name, food_scores, service_scores)

    # Prepare a final explanation message.
    explanation = (
        f"Restaurant: {restaurant_name}\n"
        f"Food Scores: {food_scores}\n"
        f"Service Scores: {service_scores}\n"
        f"Overall Score (out of 10): {overall[restaurant_name]:.3f}"
    )

    print(f"&FINAL: {explanation}")

    # TODO
    # Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
    # If you decide to use another conversation pattern, feel free to disregard this code.

    # Uncomment once you initiate the chat with at least one agent.
    # result = entrypoint_agent.initiate_chats([{}])


# DO NOT modify this code below.
if __name__ == "__main__":
    assert (
        len(sys.argv) > 1
    ), "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
