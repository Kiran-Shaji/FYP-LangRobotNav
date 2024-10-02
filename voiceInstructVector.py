"""https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"""

"""The plan is to vectorise the voice instructions and compare then with vectorised information paragraphs of the different rooms to find
the most similar room to the voice instruction. This will be an alternative to using GPT and will be fully offline."""

""""as a proof of concept, we are just gonna use pinecone and openai to go all the processing because its fast for testing purposes"""


import os
from openai import OpenAI
from pinecone import Pinecone
import time


# Initialize OpenAI API
openai_api_key = "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"

client = OpenAI(api_key = openai_api_key)
pc = Pinecone(api_key="a64697d3-921e-480d-acc6-1f55815f594f")
index = pc.Index("fyp")

# Folder containing text files
folder_path = 'roomDesc'

def embed_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Get the embedding for the text
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    return embedding

# Vectorize all text files in a folder
def embed_all_files_in_folder(folder_path):
    embeddings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            embeddings[file_name] = embed_text_file(file_path)
    
    return embeddings



def query_pinecone(text_input, namespace="text-files", top_k=2):
    # Get the embedding for the input text
    response = client.embeddings.create(
        input=text_input,
        model="text-embedding-3-large"
    )
    query_vector = response.data[0].embedding
    
    # Query Pinecone for similar vectors
    result = index.query(
        namespace=namespace,
        vector=query_vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    
    return result

def upload_embeddings_to_pinecone(embeddings, namespace="text-files"):
    vectors = []
    for file_name, embedding in embeddings.items():
        vectors.append({
            'id': file_name,  # use file name as vector ID
            'values': embedding,
            'metadata': {'file_name': file_name}
        })
    
    # Upsert vectors to Pinecone index
    index.upsert(vectors=vectors, namespace=namespace)

# Embed text files and upload them


def query_multiple_inputs(text_inputs, namespace="text-files", top_k=1):
    results = []
    counter = 0
    for text_input in text_inputs:
        counter+=1
        # Get the embedding for the input text
        time.sleep(1)
        response = client.embeddings.create(
            input=text_input,
            model="text-embedding-3-large"
        )
        query_vector = response.data[0].embedding
        
        # Query Pinecone for similar vectors
        result = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        # Collect the result
        matches = [{'input_text': text_input, 'matched_id': match['id'], 'score': match['score']} for match in result['matches']]
        results.append(matches)
        print(f"Processed {counter} out of {len(text_inputs)}")
    
    return results

# Example usage
text_inputs = [
    "Where is the sink located?",
    "Where can I find the oven?",
    "Where are the cooking utensils?",
    "Which room has the stove?",
    "Where do I go to cook food?",
    "In which room is the fridge kept?",
    "Where can I get a glass of water?",
    "Which room contains the dishwasher?",
    "Where can I prepare a meal?",
    "Where are the pots and pans stored?",
    "Where is the coffee machine?",
    "Which room has the toaster?",
    "Where do I store groceries?",
    "Where is the pantry located?",
    "Where is the kitchen counter?",
    "Which room has the cabinets for food storage?",
    "Where can I find the cutting board?",
    "Which room has the refrigerator?",
    "Where can I boil water?",
    "Which room has the blender?",
    "Where is the garbage disposal located?",
    "Where is the sink for washing dishes?",
    "Which room has the microwave?",
    "Where can I make coffee?",
    "Which room has the stove top?",
    "Where do I wash vegetables?",
    "Which room has the cooking appliances?",
    "Where can I find the tea kettle?",
    "Where is the spice rack?",
    "Where do I keep food items?"
]

text_inputs = [
    "Where is the shower located?",
    "Where can I wash my hands?",
    "Which room has the toilet?",
    "Where do I brush my teeth?",
    "Where can I take a bath?",
    "In which room is the sink for personal hygiene?",
    "Where can I find the soap and towels?",
    "Where do I go to freshen up?",
    "Which room has the mirror for grooming?",
    "Where do I store the toiletries?",
    "Where is the hair dryer kept?",
    "Which room contains the bathtub?",
    "Where do I go to use the toilet?",
    "Where can I find a towel rack?",
    "Where is the medicine cabinet located?",
    "Which room has the shower curtain?",
    "Where can I find the toothbrush?",
    "Where do I keep shampoo and conditioner?",
    "Which room has the toilet paper holder?",
    "Where can I shave?",
    "Where is the soap dispenser?",
    "Where do I hang the bath towels?",
    "Which room has the bath mat?",
    "Where can I find the hand sanitizer?",
    "Which room has the toilet seat?",
    "Where do I apply makeup?",
    "Which room has the vanity?",
    "Where can I wash my face?",
    "Where is the plunger stored?",
    "Where do I dry my hands after washing?"
]
text_inputs = [
    "Where can I sleep?",
    "Where is the bed located?",
    "Which room has the wardrobe?",
    "Where can I change clothes?",
    "Where can I find the closet?",
    "Which room has a dresser for my clothes?",
    "Where is the nightstand kept?",
    "Where do I keep my pajamas?",
    "Which room has the bedside lamp?",
    "Where can I take a nap?",
    "Where is the alarm clock?",
    "Which room has a mirror for getting dressed?",
    "Where can I store my shoes?",
    "Where can I find my pillows and blankets?",
    "Where do I fold laundry?",
    "Which room has the bed and pillows?",
    "Where do I keep my personal belongings?",
    "Where is the chest of drawers located?",
    "Which room has the bookshelf?",
    "Where can I read before going to bed?",
    "Where is the carpet in the room?",
    "Where can I find the bed frame?",
    "Which room has the curtains for blocking out light?",
    "Where do I relax and unwind at night?",
    "Where can I find extra bedding?",
    "Which room has the mattress?",
    "Where do I put my alarm clock?",
    "Where is the place to hang my clothes?",
    "Where is the room for resting?",
    "Where can I find a blanket to stay warm?"
]
text_inputs = [
    "Where is the projector set up?",
    "Where can we hold a meeting?",
    "Which room has the conference table?",
    "Where is the whiteboard located?",
    "Where can I find the video conferencing equipment?",
    "Where do we gather for discussions?",
    "Which room has the chairs arranged for a meeting?",
    "Where can I give a presentation?",
    "Where is the room for team meetings?",
    "Where can I connect my laptop for a presentation?",
    "Which room has the presentation screen?",
    "Where do we host client meetings?",
    "Where is the speakerphone for conference calls?",
    "Which room is used for brainstorming sessions?",
    "Where can I set up a projector for a meeting?",
    "Where do I find the HDMI cables for the presentation?",
    "Which room is reserved for business discussions?",
    "Where can I schedule a team meeting?",
    "Where is the large table for team meetings?",
    "Which room has the seating for presentations?",
    "Where do we hold the board meetings?",
    "Where is the presentation clicker kept?",
    "Where can I find a room for video calls?",
    "Which room has the TV screen for video conferences?",
    "Where can I meet with colleagues?",
    "Where is the microphone for presentations?",
    "Which room is used for large group discussions?",
    "Where can I find the markers for the whiteboard?",
    "Which room has the conference call system?",
    "Where do we gather for presentations?"
]

text_inputs = [
    "Where do we eat dinner?",
    "Which room has the dining table?",
    "Where can I set up the plates for a meal?",
    "Where do we gather for meals?",
    "Where is the table for family dinners?",
    "Where can I find the dining chairs?",
    "Which room has space for serving food?",
    "Where do I place the silverware for dinner?",
    "Where is the room where we have dinner parties?",
    "Where can I set up the centerpiece for the table?",
    "Which room has the chairs around the table?",
    "Where do we eat breakfast together?",
    "Where can I arrange the dinner plates?",
    "Which room is used for formal dining?",
    "Where is the room for setting up meals?",
    "Where can I serve a large meal to guests?",
    "Which room has the table for lunch?",
    "Where do I set the glasses for a meal?",
    "Where is the room with the dining set?",
    "Where can I arrange a family meal?",
    "Which room is used for hosting dinner events?",
    "Where is the cutlery set for dinner placed?",
    "Where can I find the sideboard for serving dishes?",
    "Which room is used for holiday meals?",
    "Where do I set up the buffet for a gathering?",
    "Where is the room where we host formal dinners?",
    "Which room has the chandelier above the table?",
    "Where do we sit for a group meal?",
    "Where can I serve dessert after dinner?",
    "Where do we have special occasion meals?"
]
text_inputs = [
    "Where is the sports equipment stored?",
    "Which room has the tools for maintenance?",
    "Where can I find the extra cables and wires?",
    "Where is the room with the audio-visual gear?",
    "Where can I find the projector equipment?",
    "Which room has the camera and lighting gear?",
    "Where is the storage for technical equipment?",
    "Where do we keep the workout gear?",
    "Which room holds the microphones and speakers?",
    "Where is the room with all the backup hardware?",
    "Where can I find the helmets and protective gear?",
    "Where is the sports gear stored?",
    "Which room has the tripods and camera stands?",
    "Where do we store the AV equipment?",
    "Where can I find the sound system for events?",
    "Which room holds the extra batteries and chargers?",
    "Where is the cleaning equipment stored?",
    "Where do I find the spare parts and accessories?",
    "Which room holds the lab equipment?",
    "Where is the toolbox kept?",
    "Where can I find the room with the electrical equipment?",
    "Which room has the lighting and rigging equipment?",
    "Where is the safety equipment stored?",
    "Where can I get the tools for setup?",
    "Which room contains the training equipment?",
    "Where is the backup camera gear kept?",
    "Where do we store the lab safety gear?",
    "Where is the protective clothing stored?",
    "Where can I find the sound mixing equipment?",
    "Which room holds the industrial equipment?"
]
text_inputs = [
    "Where can I find the lab equipment?",
    "Which room has the microscopes?",
    "Where do I go to conduct experiments?",
    "Where is the chemical storage located?",
    "Where can I find the safety goggles?",
    "Which room has the lab benches?",
    "Where do we analyze the samples?",
    "Where is the fume hood?",
    "Which room has the beakers and test tubes?",
    "Where can I perform scientific research?",
    "Where is the room with the lab instruments?",
    "Where do we keep the lab coats?",
    "Which room has the centrifuge?",
    "Where is the biosafety cabinet?",
    "Where can I find the autoclave?",
    "Where do I conduct research experiments?",
    "Which room has the pipettes and petri dishes?",
    "Where can I find the safety shower and eyewash station?",
    "Where do I analyze chemical reactions?",
    "Which room is used for biology experiments?",
    "Where is the lab refrigerator?",
    "Where can I find the sterile gloves?",
    "Which room has the lab scales and balances?",
    "Where do we store biological samples?",
    "Where is the incubator kept?",
    "Which room is used for testing and analysis?",
    "Where can I find the spectrometer?",
    "Where do I dispose of hazardous waste?",
    "Which room has the chemical reagents?",
    "Where can I find the gas cylinders for experiments?"
]
text_inputs = [
    "Where can I relax and watch TV?",
    "Which room has the sofa and armchairs?",
    "Where can I find the coffee table?",
    "Where is the entertainment center located?",
    "Which room has the television set?",
    "Where do we gather for family time?",
    "Where is the room with the fireplace?",
    "Where can I sit and read a book?",
    "Which room has the large windows and curtains?",
    "Where do we host guests?",
    "Where can I find the room with the couches?",
    "Which room has the bookshelves and TV stand?",
    "Where is the remote control for the TV?",
    "Where can I watch movies with the family?",
    "Which room has the rug and seating area?",
    "Where do we keep the family photo albums?",
    "Which room has the lamps and the TV?",
    "Where can I place the board games for family night?",
    "Which room has the throw pillows and blankets?",
    "Where is the stereo system located?",
    "Where do I place the decorations for the holidays?",
    "Which room is used for relaxing and socializing?",
    "Where is the home theater system set up?",
    "Which room has the comfortable seating for guests?",
    "Where can I find the recliner chair?",
    "Where do we sit and chat with friends?",
    "Which room has the coffee table books?",
    "Where can I play video games on the big screen?",
    "Where is the area for casual gatherings?",
    "Which room has the large, comfortable sofa?"
]
text_inputs = [
    "Where can I find my desk?",
    "Which room has the computer set up?",
    "Where do I go to attend meetings?",
    "Where is the printer located?",
    "Which room has the filing cabinets?",
    "Where can I find my office supplies?",
    "Where do I hold video conferences?",
    "Which room has the conference table?",
    "Where is the copy machine?",
    "Where can I find the whiteboard for brainstorming?",
    "Which room is used for individual work?",
    "Where do I keep my documents?",
    "Where can I find the stapler and tape?",
    "Which room has the phone for conference calls?",
    "Where do I go to meet clients?",
    "Where can I find the project management board?",
    "Which room has the ergonomic chairs?",
    "Where is the room for team collaborations?",
    "Where can I access the shared computer?",
    "Where is the office for quiet work?",
    "Where can I find the water cooler?",
    "Which room has the bulletin board for announcements?",
    "Where do I go for team meetings?",
    "Where can I find the laptops for presentations?",
    "Which room has the cabinets for storing supplies?",
    "Where do I keep my personal items?",
    "Where can I find the power outlets for charging?",
    "Which room is used for working on projects?",
    "Where is the area for brainstorming sessions?",
    "Where can I find the scanner?"
]
text_inputs = [
    "Where do we keep the extra office supplies?",
    "Which room has the stored paperwork?",
    "Where can I find the backup equipment?",
    "Where is the room for filing old documents?",
    "Where do we store the unused furniture?",
    "Which room has the boxes of archived files?",
    "Where can I find the extra printer paper?",
    "Where do we keep the promotional materials?",
    "Which room has the storage for old computers?",
    "Where can I find the spare office chairs?",
    "Where is the room with the shelves for storage?",
    "Where do we keep the holiday decorations?",
    "Which room holds the extra cables and accessories?",
    "Where can I find the inventory of supplies?",
    "Which room has the cabinets for storage?",
    "Where do we keep the large-format printer supplies?",
    "Where can I find the shipping materials?",
    "Where is the room for storing old files?",
    "Which room has the supplies for events?",
    "Where do we store the presentation materials?",
    "Where can I find the binders and folders?",
    "Which room is used for long-term storage?",
    "Where is the room for keeping unused equipment?",
    "Where can I find the document shredders?",
    "Which room has the containers for organizing supplies?",
    "Where do we keep the samples and prototypes?",
    "Where can I find the maintenance tools?",
    "Which room has the packing materials for shipping?",
    "Where do we store the office furniture when not in use?",
    "Where can I find the extra office decor?"
]

query_results = query_multiple_inputs(text_inputs)
for result in query_results:
    print(result)



# next we have to use chat gpt to generate a bunch of queries to test the validity of the system