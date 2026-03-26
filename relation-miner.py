from openai import OpenAI
import os
import json
import pandas as pd
from pypdf import PdfReader
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
#from structured_logprobs.main import add_logprobs

load_dotenv()

df = pd.read_excel("CEOS.xlsx")
ceos_technology = ""
instruments = []
ceos_measurements = ""
relation_examples = ""
for index, row in df.iterrows():
    if type(row["Instrument Technology"]) == str and type(row["Measurements & applications"]) == str:
        measurements = row["Measurements & applications"].split(",")
        ceos_technology += row["Instrument Technology"] + ", "
        instruments.append(row["Instrument Technology"])
        for measurement in measurements:
            if np.random.random() < 0.2:
                 relation_examples += row["Instrument Technology"] + " : " + measurement + ", "
            ceos_measurements +=  measurement + ", "
ceos_technology = ceos_technology[:-2] + "."
ceos_measurements = ceos_measurements[:-2] + "."
relation_examples = relation_examples[:-2] + "."

output = json.loads("{}")
response_schema = {
    "name":"extractions",
    "strict": True,
    "schema":{
        "type":"object",
        "properties": {
            "relations":{
                "type":"array",
                "items":{
                    "type": "object",
                    "properties": {
                        "instrument type": {"type": "string", "enum":instruments},
                        "geophysical parameter": {"type": "string"}
                        },
                    "required": ["instrument type", "geophysical parameter"],
                    "additionalProperties": False
                    }
                }
            },
        "required":["relations"],
        "additionalProperties":False
        }
    }

extraction_prompt = f"""
                    You are an assistant that takes in scientific papers and extracts two parameter types: \"instrument type\" and \"geophysical parameter\".
                    Each instrument type designates a type of sensor that can measure a geophysical parameter.
                    A single instrument type can be used to measure multiple geophysical parameters.
                    Do not extract instrument types unless they are from the following list: {ceos_technology}.
                    Examples of geophysical parameters can be found in the following list: {ceos_measurements}
                    Examples of relations can be found in the following list: {relation_examples}.
                    Use these example relations to check if the extracted geophysical parameter is reasonable given the other geophysical parameters the instrument type can measure.
                    """

postprocess_prompt = f"""
                        You are a helpful assistant that read a list of values of an instrument type and geophysical parameter pair from a json file and determines whether they are a valid pair.
                        A pair is valid if you think the given instrument type can measure the geophysical parameter and if it aligns with knowledge in the current literature, such as the CEOS database or journal papers in TGARSS or similar journals.  
                        Examples of relations from the CEOS database can be found in the following list: {relation_examples}.
                        If the pair is valid, return "yes". If the pair is not valid, return "no". Do not return anything other than "yes" or "no" specifically.
                        Complete this for every pair in the json file and separate each response unto a separate line.  
                        """
                        # f"""You are an assistant that parses through a json file and removes data that is not reasonable and readable.
                        # Return the json file that only contains the reasonable data from the input.
                        # The json contains a key of an \"instrument type\" and list of \"geophysical parameter\" values.
                        # A relation is reasonable if, for a given instrument type, the geophysical parameter can be measured by sensors that measure the other geophysical parameters found in the following list: {relation_examples}.
                        # A relation is reasonable if the instrument type can be found in the following list: {ceos_technology}.
                        # """


dani_extraction_prompt = f"""
                        You are an assistant doing information extraction from journal papers to populate a knowledge graph related to Earth observation satellites.
                        Your task is to read scientific papers and extract relations between two types of entities : \"instrument type\" and \"geophysical parameter\".
                        The only relation we care about is of type CanMeasure(instrumentType, geophysicalParameter).
                        For example, L-band radiometers (instrument type) can measure soil moisture, whereas optical imagers cannot. 
                        Return these relations as a list of values with an \"instrument type\" and a \"geophysical parameter\".
                        Each instrument type designates a type of sensor that can measure a geophysical parameter.
                        A single instrument type can be used to measure multiple geophysical parameters, and a given geophysical parameter can often be measured with different instrument types.
                        We will use the CEOS database as an implicit ontology for instrument types and geophysical parameters. 
                        Do not extract instrument types unless they are from the following list: {ceos_technology}.
                        Examples of geophysical parameters can be found in the following list: {ceos_measurements}
                        Examples of true canMeasure relations can be found in the following list: {relation_examples}.
                        Use these example relations to check if the extracted geophysical parameter is reasonable, given the other geophysical parameters the instrument type can measure.
                        Not all papers contain canMeasure relations. If a likely canMeasure relation is not found, do not return anything for that paper.
                        """

dani_postprocess_prompt = f"""
                            You are an Earth science expert doing information extraction from journal papers to populate a knowledge graph related to Earth observation satellites.
                            We are interested in relations of the type CanMeasure(InstrumentType, GeophysicalParameter). For example, L-band radiometers (instrument type) can measure soil moisture (geophysical parameter), whereas optical imagers cannot.
                            Additional examples of true canMeasure relations can be found in the following list: {relation_examples}.
                            Another agent has extracted a preliminary list of relations as a json file, and your role is to validate this list.
                            For every relation in the input, return "yes" if the relation is valid or "no" if it is invalid.
                            Do not return anything other than "yes" or "no" specifically.
                            The json contains a list of relations represented by an instrument type and geophysical parameter pair.
                            A relation is valid if it aligns with knowledge in the current literature, such as the CEOS database or journal papers in TGARSS or similar journals.  
                            Examples of relations from the CEOS database can be found in the following list: {relation_examples}
                            """
                            #f """
                            #You are an Earth science expert doing information extraction from journal papers to populate a knowledge graph related to Earth observation satellites.
                            # We are interested in relations of the type CanMeasure(InstrumentType, GeophysicalParameter).
                            # For example, L-band radiometers (instrument type) can measure soil moisture, whereas optical imagers cannot. Additional examples of true canMeasure relations can be found in the following list: {relation_examples}.
                            # Another agent has extracted a preliminary list of relations as a json file, and your role is to validate this list and remove data that seems incorrect
                            # Return a new json file that only contains the validated data from the input.
                            # The json contains a key of an \"instrument type\" and a list of \"geophysical parameter\" values.
                            # A relation is valid if it aligns with knowledge in the current literature, such as the CEOS database or journal papers in TGARSS or similar journals.  
                            # Examples of relations from the CEOS database can be found in the following list: {relation_examples}
                            # """

simple_postprocess_prompt = f"""You are an Earth Observation satellite expert giving feedback on whether a given instrument type can measure a given geophysical parameter.
                            The sentence you are giving feedback on should read - (instrument type) can measure (geophysical parameter) - for every instrument type and geophysical parameter pair.
                            The sentence is only valid if it aligns with knowledge in the current literature, such as the CEOS database or journal papers in TGARSS or similar journals and is grammatically correct.
                            Examples of pairs from the CEOS database can be found in the following list: {relation_examples}.
                            For every pair in the input, return only "yes" if the relation is valid or "no" if it is invalid.
                            """

limit = 0.01
client = OpenAI(api_key=os.getenv("API_KEY"))
for i, filename in enumerate(os.listdir("C:/Users/jamgo/OneDrive/Documents/SEAK LAB/kg-mining/papers")):
    if np.random.random() > limit: continue
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
    
    extraction = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   "role": "system", "content": dani_extraction_prompt},
            {   "role": "user", "content": paper_text}
        ],
        response_format = {"type":"json_schema", "json_schema":response_schema}
    )
    
    relations = extraction.choices[0].message.content
    relations_dict = json.loads(relations)
    postprocess_input = ""
    if len(relations_dict["relations"]) == 0: 
        continue
    else:
        for relation in relations_dict["relations"]:
            instrument = relation["instrument type"]
            parameter = relation["geophysical parameter"]
            postprocess_input += f"{instrument}:{parameter}\n"

    ## maybe change this to read the json object and take out the specific instrument types and plug that directly into the promt and only use relation_examples[<extracted_type>]
    postprocess = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   "role": "system", "content":simple_postprocess_prompt},
            {   "role": "user", "content": postprocess_input}
        ],
        logprobs = True,
        top_logprobs = 4
    )
    validity = postprocess.choices[0].logprobs.content

    try:
        prob_count = 0
        for relation in relations_dict["relations"]:
            instrument = relation["instrument type"]
            parameter = relation["geophysical parameter"]
            probability = np.exp(validity[prob_count].logprob)
            print(f"{validity[prob_count].token}({probability})-{instrument}:{parameter}")
            if validity[prob_count].token == "yes":
                if instrument in output.keys():
                    output[instrument].append(f"{parameter}:{probability}")
                else:
                    output[instrument] = [f"{parameter}:{probability}"]
            prob_count += 2
    except IndexError as error:
        print(error)
        print(relations_dict["relations"])
        print(postprocess.choices[0].message.content)
        break

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")         
output = json.dumps(output, indent=4)
with open(f"relations{time}.json", "w") as outfile:
        outfile.write(output)