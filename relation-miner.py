from openai import OpenAI # Version 1.33.0
from openai.types.beta.threads.message_create_params import Attachment, AttachmentToolFileSearch
import os
import json
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel
import numpy as np
from json.decoder import JSONDecodeError
from dotenv import load_dotenv

load_dotenv()

df = pd.read_excel("CEOS.xlsx")
ceos_technology = ""
ceos_measurements = ""
relation_examples = ""
for index, row in df.iterrows():
    if type(row["Instrument Technology"]) == str and type(row["Measurements & applications"]) == str:
        measurements = row["Measurements & applications"].split(",")
        ceos_technology += row["Instrument Technology"] + ", "
        for measurement in measurements:
            if np.random.random() < 0.1:
                 relation_examples += row["Instrument Technology"] + " : " + measurement + ", "
            ceos_measurements +=  measurement + ", "
ceos_technology = ceos_technology[:-2] + "."
ceos_measurements = ceos_measurements[:-2] + "."
relation_examples = relation_examples[:-2] + "."

json_txt = json.loads("{}")

client = OpenAI(api_key=os.getenv("API_KEY"))
for filename in os.listdir("C:/Users/jamgo/OneDrive/Documents/SEAK LAB/kg-mining/papers"):
    force_filename = "Hang et al. - 2024 - A Regionally Indicated Visual Grounding Network for Remote Sensing Images.pdf"
    #filename = force_filename

    print(filename)
    reader = PdfReader('papers/'+filename)

    paper_text = ""
    for page in reader.pages:
        page_text = page.extract_text()

        if "II" in page_text:
            page_text = page_text.split("II")[0]
            paper_text += page_text
            break

        paper_text += page_text

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {   
                "role": "system", 
                "content": """You are an assistant that takes in scientific papers to extract relations of type - <instrument type> can measure <geophysical parameter>.
                              Only extract instrument types from the following list:""" + ceos_technology +
                              """Only extract geophysical parameters from the following list:""" + ceos_measurements +
                              """Examples of relations can be found in the following - """+ relation_examples +""". 
                              Return these relations in a list of JSON objects in following format - <instrument type> : <geophysical parameter>"""
            },
            {
                "role": "user",
                "content": paper_text
            }
        ],
        response_format = { 
                            "type": "json_object" 
                          }
    )

    relation = completion.choices[0].message.content
    print(relation)
    try:
        json_temp = json.loads(relation)
        json_txt.update(json_temp)
    except JSONDecodeError:
        print(JSONDecodeError)
        continue
         

json_object = json.dumps(json_txt, indent=4)
with open("relations.json", "w") as outfile:
        outfile.write(json_object)
# def get_assistant():

#         for assistant in client.beta.assistants.list():
#             if assistant.name == 'kg miner':
#                 return assistant

#         # No Assistant found, create a new one
#         return client.beta.assistants.create(
#             model='gpt-4o',
#             description='You are a PDF parsing assistant.',
#             instructions="You are a helpful assistant designed to parse research papers. Find information from the text and files provided.",
#             tools=[{"type": "file_search"}],
#             # response_format={"type": "json_object"}, # Isn't possible with "file_search"
#             name='kg miner',
#         )

# client = OpenAI(api_key=os.getenv("API_KEY"))

# for filename in os.listdir("C:/Users/jamgo/OneDrive/Documents/SEAK LAB/kg-mining/papers"):

#     # Upload pdf(s) to the OpenAI API
#     file = client.files.create(
#         file=open('papers\\'+filename, 'rb'),
#         purpose='assistants'
#     )

#     # Create thread
#     thread = client.beta.threads.create()

#     # Create an Assistant (or fetch it if it was already created). It has to have
#     # "file_search" tool enabled to attach files when prompting it.

#     # Add your prompt here
#     prompt = "Extract relations of type: <instrument type> can measure <geophysical parameter>. Output only the the relations in a json format, such as instrument:parameter."
#     client.beta.threads.messages.create(
#         thread_id = thread.id,
#         role='user',
#         content=prompt,
#         attachments=[Attachment(file_id=file.id, tools=[AttachmentToolFileSearch(type='file_search')])]
#     )

#     # Run the created thread with the assistant. It will wait until the message is processed.
#     run = client.beta.threads.runs.create_and_poll(
#         thread_id=thread.id,
#         assistant_id=get_assistant().id,
#         timeout=300, # 5 minutes
#         # response_format={"type": "json_object"}, # Isn't possible
#     )

#     # Eg. issue with openai server
#     if run.status != "completed":
#         raise Exception('Run failed:', run.status)

#     # Fetch outputs of the thread
#     messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
#     messages = [message for message in messages_cursor]

#     message = messages[0] # This is the output from the Assistant (second message is your message)
#     assert message.content[0].type == "text"

#     # Output text of the Assistant
#     res_txt = message.content[0].text.value
#     print(res_txt)

#     res_txt = res_txt[6:]
#     res_txt = res_txt[:-3]
#     res_txt = res_txt[:res_txt.rfind('}')+1]
#     res_txt = res_txt[res_txt.find('{'):]
#     res_txt.strip()

#     json_object = json.dumps(json.loads(res_txt))
#     with open("relations.json", "w") as outfile:
#         outfile.write(json_object)

#     # # Delete the file(s) afterward to preserve space (max 100gb/company)
#     client.files.delete(file.id)