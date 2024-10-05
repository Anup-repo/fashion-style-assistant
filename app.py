import streamlit as st
from PIL import Image
from create_db import create_databases
from retriver import retrieve_results

def main():
    st.title("Fashion Style Assistant")

    user_query = st.text_input(
        "Enter your fashion query", placeholder="e.g., How to style a red dress?"
    )

    if st.button("Get Styling Ideas"):
        if user_query:
            if create_databases():
                st.write("vector database exist. so retriving the data")
            st.write("Fetching top 3 images and generating response...")

            response, results = retrieve_results(user_query)
            images = []
            for uri in results["uris"][0][:3]:
                image = Image.open(uri)
                st.image(image, caption=f"Image from {uri}")
                with open(uri, "rb") as img_file:
                    images.append(img_file.read())

            st.write("Styling Suggestions:")
            st.write(response)  # Adjust as per API response structure
        else:
            st.error("Please enter a query.")

if __name__ == '__main__':
    main()
