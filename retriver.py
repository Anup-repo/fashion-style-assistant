import os
import base64
import logging
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up logging configuration for production-level logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


# Function to initialize the llama model from Together
def initialize_gpt_model(api_key):
    """
    Initialize the gpt-4o with the specified API key and temperature.
    Args:
        api_key: The API key for LLM.
    Returns:
        An instance of the GPT-4o model.
    """
    gpt4o = ChatOpenAI(model="gpt-4o", temperature=0.0,api_key=api_key)
    return gpt4o


# Function to define and create the chat prompt
def create_chat_prompt():
    """
    Create the ChatPromptTemplate for vision-based query using image context.
    Returns:
        A ChatPromptTemplate object with predefined system and user messages.
    """
    image_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful fashion and styling assistant. You will provide styling advice by focusing on the details in the images.",
            ),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": "{user_query}",
                    },
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{image_data_1}",
                    },
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{image_data_2}",
                    },
                ],
            ),
        ]
    )
    return image_prompt


# Function to encode image as base64
def encode_image_to_base64(image_path):
    """
    Convert an image to base64 encoding.
    Args:
        image_path: Path to the image file.
    Returns:
        Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode("utf-8")


# Function to format the input prompt
def format_prompt_inputs(data, user_query):
    """
    Prepare the prompt inputs for the chain by encoding image files and adding user query.
    Args:
        data: The dictionary containing image URIs from the database.
        user_query: The text-based query from the user.
    Returns:
        A dictionary with formatted inputs for the prompt.
    """
    inputs = {}
    inputs["user_query"] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # Encode the images
    inputs["image_data_1"] = encode_image_to_base64(image_path_1)
    inputs["image_data_2"] = encode_image_to_base64(image_path_2)

    return inputs


# Function to initialize the image vector database (ChromaDB)
def load_image_vdb():
    """
    Load or create the image vector database (ChromaDB) and return the collection.
    Returns:
        The image vector database collection.
    """
    chroma_client = chromadb.PersistentClient(path="image_vdb")
    image_loader = ImageLoader()

    # Initialize OpenCLIP for image embeddings
    CLIP = OpenCLIPEmbeddingFunction()

    # Create or load the collection
    image_vdb = chroma_client.get_collection(
        name="image", embedding_function=CLIP, data_loader=image_loader
    )

    return image_vdb


# Function to query the image vector database
def query_db(query, results=3):
    """
    Query the image vector database to retrieve similar images.
    Args:
        query: The query text provided by the user.
        results: Number of similar images to retrieve (default: 3).
    Returns:
        A dictionary containing the URIs and distances of the retrieved images.
    """
    # Load the vector database
    image_vdb = load_image_vdb()

    # Perform the query
    results = image_vdb.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results


# Main function to execute the vision chain
def execute_vision_chain(api_key, user_query):
    """
    Execute the vision chain by retrieving images from the vector database, formatting the prompt,
    and invoking the gpt-4o model.
    Args:
        api_key: The API key for gpt.
        user_query: The text query from the user for styling suggestions.
    Returns:
        The response from the vision chain.
    """
    # Initialize Llama model
    gpt4o = initialize_gpt_model(api_key)

    # Define the prompt template and output parser
    image_prompt = create_chat_prompt()
    parser = StrOutputParser()

    # Create the LangChain chain
    vision_chain = image_prompt | gpt4o | parser

    # Query the image database for relevant images
    results = query_db(user_query, results=3)

    # Format the prompt inputs
    prompt_input = format_prompt_inputs(results, user_query)

    # Invoke the chain with formatted inputs
    response = vision_chain.invoke(prompt_input)

    return response, results


def retrieve_results(user_query):
    # API key for Together
    API_KEY = os.getenv("API_KEY")

    if not API_KEY:
        logging.error(
            "API key for  Chat GPT is missing. Please set the API_KEY environment variable."
        )
    else:
        # Execute the vision chain and get the response
        response, image_result = execute_vision_chain(API_KEY, user_query)
        return response, image_result
