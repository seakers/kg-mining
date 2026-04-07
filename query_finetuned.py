from dotenv import load_dotenv

import os
import random
import pickle
import time
import requests
import json
import re

load_dotenv()

def answer_queries(model_name, filename, sample_size):
    
    main =  """
                You will be given a question that describes an instrument and asks if the instrument is capable of performing a measurement.
                If one of the main purposes of this instrument is to perform the measurement, we will categorize it as "Primary".
                If this instrument can perform the measurement but is not best suited for the job or is not commonly used, we will categorize it as "Secondary".
                If this instrument cannot be used to perform the measurement, we will categorize it as "Tertiary".
                Answer the question by responding with the appropriate category label for the instrument with an explanation of your response afterwards given your prior knowledge and the context provided.
                Give a confidence score from 0 to 100 based on how certain you are about the answer you are providing.
            """

    responses = {}

    try:
        with open(f"responses/{filename}.pkl", "rb") as file:
            responses = pickle.load(file)

    except FileNotFoundError:
        print(f"No current {filename}.pkl in responses/ folder available to load.")

    answered_queries = responses.keys()
    with open("queries.txt", "r", encoding='utf-8') as file:
        content = file.read()
    
    lines = content.split("A sensor is of type")
    lines = random.sample(lines, sample_size)

    start = time.time()
    print(f"Starting inference for finetuned model for {model_name}. Number of queries = {sample_size}")
    try:
        for line in lines:

            inter_start = time.time()
            # if inter_start - start > 3600:
            #     print("TIMEOUT - started queries at {start}, this query started at {inter_start}")
            #     break

            query="A sensor is of type "+line.strip()
            messages = [("system",f"You are an Earth Science expert validating the capabilities of measurement devices on Earth Observation satellites. {main}"),
                        ("human", f"Question:{query}"),]

            if query not in answered_queries:
                response = stream_chat_completion(messages)
                responses[query] = responses
                
            
            print(f"Time for query: {time.time()-inter_start}")

    except KeyboardInterrupt:
        print(f"User interupted RAG/LLM query.")

    print(f"Saving stored responses to responses/{filename}.pkl")
    with open(f"responses/{filename}.pkl", "wb") as file:
        pickle.dump(responses,file)

def stream_chat_completion(messages, api_url=None):
    """
    Call the streaming chat completion API

    Parameters:
        access_token (str): Authentication token
        user_message (str): User message content
        api_url (str, optional): API URL, defaults to GeoGPT service

    Returns:
        list: List containing all response chunks
    """
    # Set default API URL
    if api_url is None:
        api_url = "https://geogpt-sg.zero2x.org/be-api/service/api/model/v1/chat/completions"

    # Prepare request headers and data
    headers = {
        'Authorization': f'Bearer {os.getenv("GEOGPT_API_KEY")}',
        "Content-Type": "application/json"
    }

    payload = {
        "messages": messages,
        "stream": True
    }

    responses = []  # Store all response chunks

    try:
        # Send POST request (enable streaming response)
        with requests.post(api_url, headers=headers, json=payload, stream=True) as response:
            # Check response status
            response.raise_for_status()

            print(f"Response status code: {response.status_code}")
            print("Starting to receive streaming response...")

            # Process streamed responses
            for chunk in response.iter_lines():
                # Filter out keep-alive new lines
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')

                    try:
                        # Handle possible Server-Sent Events (SSE) format
                        if decoded_chunk.startswith("data:"):
                            json_str = decoded_chunk[5:]
                        else:
                            json_str = decoded_chunk

                        # Check for message event tag
                        if json_str == 'event:message':
                            continue
                        # Check for end flag
                        if json_str.strip() == "[DONE]":
                            print("\nReceived end flag [DONE]")
                            break
                            
                        # Unescape handling:
                        # 1. Replace \" with "
                        # 2. Replace \\ with \
                        unescaped_str = json_str.replace('\\"', '"').replace('\\\\', '\\')

                        if unescaped_str.startswith('"') and unescaped_str.endswith('"'):
                            # Remove outer quotes
                            unescaped_str = unescaped_str[1:-1]

                        # Parse JSON
                        data = json.loads(unescaped_str)
                        responses.append(data)  # Save response

                        # Extract and print content
                        if 'choices' in data and data['choices']:
                            choice = data['choices'][0]
                            # Check if delta field exists
                            if 'delta' in choice:
                                content = choice['delta'].get('content', '')
                                if content:
                                    print(content, end='', flush=True)

                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse JSON: {decoded_chunk}")
                        print(f"Error details: {str(e)}")
                    except Exception as e:
                        print(f"\nProcessing error: {str(e)}")

    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {str(e)}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

    return responses


def read_responses(filename):

    responses = {}
    with open(f"responses/{filename}.pkl", "rb") as file:
        responses = pickle.load(file)

    incorrect = 0
    total = 0
    for query in responses.keys():

        total += 1

        response = responses['choices'][0].get('delta', {}).get('content', '')

        if "tertiary" in response.lower()[:100]:
            incorrect += 1
            #print(f"Query - {query}")
            #print(responses[query]["context"])
            #print(f"RAG Response - {responses[query]['answer']}")
        
            #print("-"*225)
    
    print(f"Incorrect finetuned responses = {incorrect}/{total}")


if __name__ == "__main__":

    model_name = "geogpt-qwen3:32b"
    #model_name = "gpt-5"
    filename = model_name.replace(":", "-")
    print(f"Using responses/{filename}.pkl")

    sample_size=2

    answer_queries(model_name, filename, sample_size)
    read_responses(filename)