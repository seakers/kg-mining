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
            if np.random.random() < 0.2:
                 relation_examples += row["Instrument Technology"] + " : " + measurement + ", "
            ceos_measurements +=  measurement + ", "
ceos_technology = ceos_technology[:-2] + "."
ceos_measurements = ceos_measurements[:-2] + "."
relation_examples = relation_examples[:-2] + "."

json_txt = json.loads("{}")

client = OpenAI(api_key=os.getenv("API_KEY"))
for i, filename in enumerate(os.listdir("C:/Users/jamgo/OneDrive/Documents/SEAK LAB/kg-mining/papers")):
#for filename in ["Hang et al. - 2024 - A Regionally Indicated Visual Grounding Network for Remote Sensing Images.pdf"]:
    if i == 150: break

    reader = PdfReader('papers/'+filename)

    paper_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        
        if "II" in page_text:
            page_text = page_text.split("II")[0]
            paper_text += page_text
            break

        paper_text += page_text

    extraction = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   
                "role": "system", 
                "content": f"""You are an assistant that takes in scientific papers to extract relations with two parameters: \"instrument type\" and \"geophysical parameter\".
                              Return these relations as a list of JSON objects in following format - \"(instrument type)\" : \"(geophysical parameter)\".
                              Each instrument type designates a type of sensor that can measure a geophysical parameter.
                              Do not extract instrument types unless they are from the following list: {ceos_technology}.
                              Examples of geophysical parameters can be found in the following list: {ceos_measurements}
                              Examples of relations can be found in the following list: {relation_examples}.
                              Use these examples relations to check if the extracted geophysical parameter is reasonable giving other geophysical parameters the instrument type can measure.
                              If a reasonable relation is not found, return "N/A : N/A"
                              """
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

    ## maybe change this to read the json object and take out the specific instrument types and plug that directly into the promt and only use relation_examples[<extracted_type>]
    post_process = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   
                "role": "system", 
                "content": f"""You are an assistant that parses through a json file and removes data that is not reasonable and readable.
                               Return the a json file that only contains the reasonable data from the input.
                               The json contains relations with two parameters: \"instrument type\" and \"geophysical parameter\".
                               A relation is reasonable if, for a given instrument type, the geophysical parameter can be measured by sensors that measure the other geophysical parameters found in the following list: {relation_examples}.
                               A relation is readable if the instrument type can be found in the following list: {ceos_technology}.
                              """
            },
            {
                "role": "user",
                "content": extraction.choices[0].message.content
            }
        ],
        response_format = { 
                            "type": "json_object" 
                          }
    )

    relation = post_process.choices[0].message.content
    try:
        # unfiltered = json.loads(post_process.choices[0].message.content)
        
        # post_process = client.beta.chat.completions.parse(
        #     model="gpt-4o",
        #     messages=[
        #         {   
        #             "role": "system", 
        #             "content": f"""You are an assistant that parses through a json file and removes data that is not reasonable and readable.
        #                         Return the a json file that only contains the reasonable data from the input.
        #                         The json contains relations with two parameters: \"instrument type\" and \"geophysical parameter\".
        #                         A relation is reasonable if, for a given instrument type, the geophysical parameter can be inferred from the other geophysical parameters found in the following list: {relation_examples}.
        #                         A relation is readable if the instrument type can be found in the following list: {ceos_technology}.
        #                         """
        #         },
        #         {
        #             "role": "user",
        #             "content": relations
        #         }
        #     ],
        #     response_format = { 
        #                         "type": "json_object" 
        #                     }
        # )
        json_temp = json.loads(relation)
        json_txt.update(json_temp)
    except JSONDecodeError:
        print(JSONDecodeError)
        continue
         

json_object = json.dumps(json_txt, indent=4)
with open("relations.json", "w") as outfile:
        outfile.write(json_object)