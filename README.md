# Fashion Style Assistant - Streamlit App

This Streamlit app allows users to input fashion-related queries and receive styling suggestions based on image context. The app uses a vector database of images, powered by ChromaDB and OpenCLIP, and generates fashion suggestions using the GPT-4o model.

## Features

- Accepts a user query related to fashion styling.
- Retrieves the top 3 most relevant images from a vector database of fashion images.
- Displays the images and provides personalized fashion advice.
- Supports easy setup and API integration.

## Installation

### Prerequisites

- Python 3.8+
- Create and activate a virtual environment:
  
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Steps

**Step 1: Clone the Repository**
```bash
git clone https://github.com/Anup-repo/fashion-style-assistant.git
cd fashion-style-assistant
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: API Key Setup**
You need to store your Together AI API key in a .env file.

Create a ``` .env``` file in the root of your project directory.

Add your API key to the file as follows:

```bash
API_KEY=your-gpt-4o-ai-api-key
```

**Step 4: Create or Download the Fashion Data**

You can either create the fashion_data folder by running the app or use the zipped dataset.

*Option 1: Generate fashion_data via Streamlit*
Simply run the Streamlit app, and it will download and create the fashion_data folder using 2,000 images from the Fashionpedia dataset.

```bash
streamlit run app.py
```
The images will be stored in the fashion_data directory automatically.

*Option 2: Use Pre-zipped fashion_data*
Download the zipped fashion_data folder https://drive.google.com/file/d/1Uvl6g3Ck2XswzANlXVfKsNb_tjIkqFip/view?usp=sharing.

Unzip the fashion_data folder into the root directory into ```fashion_data``` folder:

**Step 5: Run the Application**
Once you've set up your environment and fashion data, you can start the Streamlit app:

```bash
streamlit run app.py
```
It will create a vector dataabse called ```image_vdb``` and then gpt will be used to retirve the content.

### Project Structure

├── fashion_data/             # Directory containing fashion images (created/filled during app run)    
├── image_vdb/                # chroma vector database folder    
├── app.py                    # Main Streamlit app script    
├── create_db.py              # Create vector database   
├── retriver.py               # retirive content from database   
├── requirements.txt          # Python dependencies   
├── .env                      # File to store API key   
└── README.md                 # Project documentation
