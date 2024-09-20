# CHATBOT
If you are using VS Code or any other code editor:
Create GROQ_API_KEY, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT (project name) in a .env file.
Create a virtual environment using Python 3.10.
Run the following commands in the terminal:
pip install -r requirements.txt,
streamlit run app.py.
I have also provided a .ipynb file, which I used in Google Colab. For this code to work:
Add the API keys(GROQ_API_KEY, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT) to the secrets.
!pip install -r requirements.txt
Run the following command to get an IP address:
!wget -q -O - ipv4.icanhazip.com,
After getting the IP address, run:
!streamlit run app.py & npx localtunnel --port 8501,
When streamlit app runs enter your query
