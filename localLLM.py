from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_llm_responses(input_texts, system_prompt, max_new_tokens=2):
    """
    Takes a list of input texts and generates responses using the Qwen LLM, including a system prompt.

    Args:
        input_texts (list): A list of input text strings to query the LLM.
        system_prompt (str): The system prompt that guides the model's responses.
        max_new_tokens (int): The maximum number of tokens to generate in the response.
    
    Returns:
        list: A list of generated responses from the LLM.
    """
    responses = []
    
    for text in input_texts:
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Convert messages to the required input format for the model
        start_time = time.time()
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize the input
        model_inputs = tokenizer([model_inputs], return_tensors="pt").to(model.device)

        # Generate the response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        elapsed_time = time.time() - start_time
        print(f"Response generated in {elapsed_time:.2f} seconds")
        # Decode the generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
    
    return responses

# Example usage
input_queries = [
    "Where can I find my desk?",
    "Which room has the computer set up?",
    "Where do I go to attend meetings?",
]

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
# Call the function and print the responses
llm_responses = get_llm_responses(input_queries, system_prompt)
for i, response in enumerate(llm_responses):
    print(f"Response for '{input_queries[i]}': {response}")
