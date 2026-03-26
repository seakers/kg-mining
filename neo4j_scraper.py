from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
import regex as re
import pickle as pkl
import random
import os
from dotenv import load_dotenv

load_dotenv()

def store_numerical_properties(driver):
    numerical_properties = {}
    property_sums = {}
    property_keys = set()

    numerical_summaries = {}
    summary_keys = set()
    embedding_model = OpenAIEmbeddings(api_key=os.getenv("API_KEY"), model="text-embedding-3-small")

    with driver.session() as session:
        pattern = r'-?\d+\.?\d*'
        prop_result = session.run(
                """
                MATCH (n)
                RETURN id(n) AS id, properties(n) AS props
                """
            )

        for record in prop_result:
            properties = record["props"]
            node_properties = {}
            node_summaries = {}

            for key, value in properties.items():
                if isinstance(value, (int,float)):
                    node_properties[key] = value
                    property_keys.add(key)

                    if key in property_sums.keys():
                        property_sums[key] = property_sums[key] + [float(value)]
                    else:
                        property_sums[key] = [float(value)]

                elif key in ["orbit_period", "orbit_longitude", "max_swath", "best_resolution"]:
                    numbers = re.findall(pattern, value)
                    if len(numbers) > 0:
                        number = [float(x) if "." in value else int(x) for x in numbers][0]

                        node_properties[key] = number
                        property_keys.add(key)

                        if key in property_sums.keys():
                            property_sums[key] = property_sums[key] + [float(number)]
                        else:
                            property_sums[key] = [float(number)]
                
                elif "summary" in key:
                    node_summaries[key] = value
                    summary_keys.add(key)
                    
            
            if node_properties:
                numerical_properties[record["id"]] = node_properties
            if node_summaries:
                numerical_summaries[record["id"]] = node_summaries
        
        property_averages = {}
        for key in property_sums.keys(): property_averages[key] = sum(property_sums[key])/len(property_sums[key])
        
        summary_embeddings = {}
        summary_mask = {}
        for node in numerical_summaries.keys():
            text = []
            mask = []

            for key in summary_keys:
                if key in numerical_summaries[node].keys():
                    text.append(numerical_summaries[node][key])
                    mask.append(1)
                else:
                    mask.append(0)

            if text: embeddings = embedding_model.embed_documents(text)
            demb = embedding_model.dimensions
            demb = 1536

            for i, value in enumerate(mask):
                if value == 0:
                    embeddings.insert(i, [0]*demb)
            summary_embeddings[node] = embeddings
            summary_mask[node] = mask
        
        with open("attributes.pkl", "wb") as file: pkl.dump(numerical_properties, file)
        with open("averages.pkl", "wb") as file: pkl.dump(property_averages, file)
        with open("vocab.pkl", "wb") as file: pkl.dump(list(property_keys), file)

        with open("summary_embeddings.pkl" , "wb") as file: pkl.dump(summary_embeddings, file)
        with open("summary_masks.pkl" , "wb") as file: pkl.dump(summary_mask, file)

def store_triplets(driver):
    triplets = []
    train_str = ""
    test_str = ""
    valid_str = ""

    with driver.session() as session:

        triplet_result = session.run(
                """
                MATCH (s)-[r]->(o)
                RETURN id(s) AS subject,
                       type(r) AS predicate,
                       id(o) AS object
                """
            )

        for record in triplet_result:
            triplet = (
                record["subject"],
                record["predicate"],
                record["object"]
            )
            triplets.append(triplet)

        with open("triplets.txt", "w") as file:
            for triplet in triplets:
                triplet_to_str = [str(t) for t in triplet]
                triplet_str = "\t".join(triplet_to_str)+"\n"
                file.write(triplet_str)

                if random.random() < 0.9:
                    train_str += triplet_str
                elif random.random() < 0.95:
                    test_str += triplet_str
                else:
                    valid_str += triplet_str
        
        with open("train.txt", "w") as file: file.write(train_str)
        with open("test.txt", "w") as file: file.write(test_str)
        with open("valid.txt", "w") as file: file.write(valid_str)

def make_queries(driver):
    with driver.session() as session:
        query_result = session.run(
                """
                MATCH (a:SensorType)-[:PARENT_OF]->(b:Sensor)-[r:OBSERVES]->(c:ObservableProperty)
                RETURN a.name as type, properties(b) as props, c.name as observation
                """
            )
        
        with open("queries.txt", "w", encoding='utf-8') as file:
            for data in [record.data() for record in query_result]: 
                entity_properties = data["props"]
                max_swath = ""
                best_resolution = ""
                resolution_summary = ""
                wavebands = ""
                waveband_summary = ""
                accuracy_summary = ""

                if "max_swath" in entity_properties.keys():
                    max_swath = f"The maximum swath is {entity_properties['max_swath']}. "
                if "best_resolution" in entity_properties.keys():
                    best_resolution = f"The best resolution the sensor can achieve is {entity_properties['best_resolution']}. "
                    if "resolution_summary" in entity_properties.keys():
                        resolution_summary = f"A detailed overview of the resolution is as follows: {entity_properties['resolution_summary']}. "
                if "wavebands" in entity_properties.keys():
                    if len(entity_properties["wavebands"]) > 1:
                        beginning = ", ".join(entity_properties["wavebands"][:-1])
                        end = ", and "+ entity_properties["wavebands"][-1]
                        wavebands = f"The sensor can record in the {beginning}{end} wavebands. "
                    else:
                       wavebands = f"The sensor can record in the {entity_properties['wavebands'][0]} waveband. " 
                    if "waveband_summary" in entity_properties.keys():
                        waveband_summary = f"A detailed overview of the ranges for each waveband is as follows: {entity_properties['waveband_summary']}."
                if "accuracy_summary" in entity_properties.keys():
                    accuracy_summary = f"A summary of the accuracy is as follows: {entity_properties['accuracy_summary']}. "
                

                sensor_type = data["type"]
                observation_type = data["observation"]
                file.write(f"A sensor is of type {sensor_type}.{max_swath}{best_resolution}{resolution_summary}{wavebands}{waveband_summary}{accuracy_summary} Can the described sensor measure {observation_type}?\n")

def main(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    store_numerical_properties(driver)
    #store_triplets(driver)
    #make_queries(driver)
    driver.close()

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "ceosdb_scraper"
    main(uri, user, password)