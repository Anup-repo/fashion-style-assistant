import os
import logging
import chromadb
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Set up logging configuration for production-level logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Define the folder where images will be saved
destination_path = "fashion_data"


# Check if the dataset folder exists, if not, download the dataset
def download_dataset_if_needed(destination_folder):
    """
    Check if the 'fashion_data' folder exists.
    If not, log the message, download the dataset, and save the images.
    If it exists, log the message that the folder is found.
    """
    if not os.path.exists(destination_folder):
        logging.info(
            f"'{destination_folder}' folder does not exist. Downloading dataset from Hugging Face."
        )
        os.makedirs(destination_folder, exist_ok=True)

        # Download the dataset from Hugging Face
        dataset = load_dataset("detection-datasets/fashionpedia")

        # Save the first 2000 images from the dataset
        save_images(dataset, destination_folder, num_images=2000)
        logging.info(f"Saved the first 2000 images to {destination_folder}")
    else:
        logging.info(
            f"'{destination_folder}' folder exists. Proceeding to create the vector database."
        )


def save_images(dataset, dataset_folder, num_images=1000):
    """
    Save the first num_images from the Hugging Face dataset to the local directory.
    Args:
        dataset: The dataset object from Hugging Face.
        dataset_folder: Folder where the images will be saved.
        num_images: The number of images to save.
    """
    for i in range(num_images):
        image = dataset["train"][i]["image"]
        image.save(os.path.join(dataset_folder, f"image_{i+1}.png"))


def create_chromadb(db_name="image_vdb"):
    """
    Create a persistent ChromaDB client and initialize the image vector database.
    Logs messages about database creation.
    """
    chroma_client = chromadb.PersistentClient(path=db_name)

    # Log that the ChromaDB is being created with the images
    logging.info("Creating image vector database using the first 2000 images...")

    image_loader = ImageLoader()

    # Initialize OpenCLIP for image embeddings
    CLIP = OpenCLIPEmbeddingFunction()

    # Create or retrieve an existing collection
    image_vdb = chroma_client.get_or_create_collection(
        name="image", embedding_function=CLIP, data_loader=image_loader
    )
    return image_vdb


def add_batch(image_vdb, batch_ids, batch_uris):
    """
    Add a batch of images to the ChromaDB vector database.
    Args:
        image_vdb: The image vector database.
        batch_ids: The list of image IDs.
        batch_uris: The list of image paths (URIs).
    """
    image_vdb.add(ids=batch_ids, uris=batch_uris)


def process_images_in_batches(destination_folder, batch_size, limit):
    """
    Process images in batches to add to the database.
    Args:
        destination_folder: The folder where images are stored.
        batch_size: Number of images per batch to add to the database.
        limit: The total number of images to process.
    Yields:
        batch_ids: A list of image IDs for the batch.
        batch_uris: A list of image URIs for the batch.
    """
    ids = []
    uris = []
    count = 0

    # Collect image paths and ids
    for i, filename in enumerate(sorted(os.listdir(destination_folder))):
        if filename.endswith(".png"):
            file_path = os.path.join(destination_folder, filename)
            ids.append(str(i))
            uris.append(file_path)
            count += 1

        # When batch is ready or limit is reached, process it
        if len(ids) >= batch_size or count >= limit:
            yield ids, uris
            ids, uris = [], []  # Reset for the next batch

        # Stop processing if the limit is reached
        if count >= limit:
            break

    # Process any remaining images after the loop
    if ids and uris:
        yield ids, uris


def create_databases():
    if os.path.exists("image_vdb"):
        return True
    # Step 1: Check if the folder exists and download if needed
    download_dataset_if_needed(destination_path)

    # Step 2: Create ChromaDB and add images
    image_vdb = create_chromadb()

    batch_size = 1000
    limit = 2000

    # Log that we're starting to add the images
    logging.info(f"Adding the first {limit} images to the vector database...")

    # Step 3: Process the images in batches and add them to the database
    with ThreadPoolExecutor() as executor:
        for batch_ids, batch_uris in process_images_in_batches(
            destination_path, batch_size, limit
        ):
            executor.submit(add_batch, image_vdb, batch_ids, batch_uris)

    logging.info(f"Successfully added the first {limit} images to the vector database.")
