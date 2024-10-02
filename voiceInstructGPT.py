import os
from openai import OpenAI
import whisper
import time
# Set up your OpenAI API key
client = OpenAI(
    api_key="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")

# Load the Whisper model
model = whisper.load_model("base")  # You can use "tiny", "base", etc.

# Function to transcribe audio to text using Whisper
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

# Function to send transcribed text to OpenAI GPT model with a system prompt
def send_to_gpt_with_system_prompt(transcribed_text, system_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # You can also use "gpt-3.5-turbo" here
        messages=[
            {"role": "system", "content": system_prompt},  # System prompt that guides the model
            {"role": "user", "content": transcribed_text}  # User prompt (transcribed audio)
        ],
        max_tokens=150,  # Adjust based on desired response length
        temperature=0.7,  # Adjust creativity
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return completion.choices[0].message.content.strip()

def get_responses(input_texts, system_prompt):
    """
    Takes a list of input texts, sends them to the GPT model using the provided system prompt,
    and prints out the responses.

    Args:
        input_texts (list): A list of input text strings to query the GPT model.
        system_prompt (str): The system prompt to guide the model's responses.
    """
    responses = []
    counter = 0
    for text in input_texts:
        gpt_response = send_to_gpt_with_system_prompt(text, system_prompt)
        responses.append(gpt_response)
        time.sleep(1)
        counter += 1
        print(f"Processed {counter}/{len(input_texts)} inputs")
    # Print out the responses
    for i, response in enumerate(responses):
        print(f"Response for '{input_texts[i]}': {response}")


# Example usage
if __name__ == "__main__":
    # Transcribe your audio file
    audio_file = "test1.m4a"  # Path to your audio file
    #transcribed_text = transcribe_audio(audio_file)

    # Define your system prompt
    system_prompt = """You are an AI assistant that helps direct a user to a specific location. Based on the following locations, pick the one that best fits the user's request. Return only the name of the location.
    Locations: 
    Living Room
    Kitchen
    Bedroom
    Bathroom
    Dining Room
    Office
    Storage Room
    Laboratory
    Conference Room
    Equipment Room"""

    # Print the transcribed text
    #print("Transcribed Text:", transcribed_text)

    # Send the transcribed text to GPT along with the system prompt and get the response
    #gpt_response = send_to_gpt_with_system_prompt(transcribed_text, system_prompt)

    # Output the GPT model's response
    #print("GPT Response:", gpt_response)
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
    text_inputs1 = [
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


    get_responses(text_inputs, system_prompt)
    print("------------------------------------------------")
    get_responses(text_inputs1, system_prompt)


    
