import os
import re
import ast
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import mysql.connector
from mysql.connector import Error
import ollama
from ollama import ResponseError
import chromadb
import time
import random
import pandas as pd
import numpy as np

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Settings
DB_PARAMS = {
    "host": "192.168.2.116",
    "database": "memory_agent",
    "user": "teslavortex",
    "password": "kittypop",
    "port": 3306,
}

OLLAMA_CLIENT = ollama.Client(host="192.168.2.76")
CHROMADB_CLIENT = chromadb.Client()

# NLP Tools
sentiment_analyzer = SentimentIntensityAnalyzer()


class Memory:
    def __init__(self, db_name, persist_directory):
        self.db_name = db_name
        self.persist_directory = persist_directory
        self.db_path = os.path.join(persist_directory, f"{db_name}.pkl")
        self.client = ollama.Client(host="192.168.2.76")


    def load_db(self):
        """Load existing database"""
        df = pd.read_pickle(self.db_path)
        print(f"Markdown database loaded with {len(df)} records.")
        return df


    def create_db(self, doc_dir):
        """Create database from markdown files in doc_dir"""
        embed_db = []
        for filename in os.listdir(doc_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(doc_dir, filename)
                with open(file_path, 'r') as file:
                    markdown_text = file.read()
                    chunks = self.extract_headings(markdown_text)
                    for chunk in chunks:
                        embed_db.append({
                            'File_Name': filename,
                            'Title': chunk['Title'],
                            'Text': chunk['Text']
                        })
        df = pd.DataFrame.from_records(embed_db)
        df['Embeddings'] = df.apply(lambda row: self.generate_embedding(row['Title'], row['Text']), axis=1)
        df.to_pickle(self.db_path)
        print(f"Database created with {len(df)} records.")


    def update_db(self, doc_dir):
        """Update database with new markdown files in doc_dir"""
        existing_files = set(self.load_db()['File_Name'].tolist())
        new_files = [filename for filename in os.listdir(doc_dir) if filename.endswith('.md') and filename not in existing_files]

        if new_files:
            print(f"Found new files: {new_files}")
            for filename in new_files:
                file_path = os.path.join(doc_dir, filename)
                with open(file_path, 'r') as file:
                    markdown_text = file.read()
                    new_records = self.extract_headings(markdown_text)
                    new_df = pd.DataFrame.from_records([
                        {'File_Name': filename, 'Title': record['Title'], 'Text': record['Text']}
                        for record in new_records
                    ])
                    new_df['Embeddings'] = new_df.apply(lambda row: self.generate_embedding(row['Title'], row['Text']), axis=1)
                    existing_df = self.load_db()
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                    updated_df.to_pickle(self.db_path)
            print("Database updated successfully.")
        else:
            print("No new files found.")



    def extract_headings(self, markdown_text):
        """
        Extract headings from markdown text.

        Args:
        - markdown_text (str): Markdown text to extract headings from.

        Returns:
        - chunks (list[dict]): List of dictionaries containing title and text.
        """
        lines = markdown_text.split('\n')
        chunks = []
        current_path = []
        content = []
        in_code_block = False

        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                content.append(line)
            elif in_code_block:
                content.append(line)
            else:
                match = re.match(r'^(#{1,6})\s*(.+)', line)
                if match:
                    if content:
                        chunks.append({
                            'Title': '/'.join(current_path),
                            'Text': '\n'.join(content).strip()
                        })
                        content = []
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    current_path = current_path[:level] + [title]
                elif line.strip():
                    content.append(line)

        if content:
            chunks.append({
                'Title': '/'.join(current_path),
                'Text': '\n'.join(content).strip()
            })

        return chunks


    def generate_embedding(self, title, text):
        """
        Generate embedding for title and text.

        Args:
        - title (str): Title of the text.
        - text (str): Text to generate embedding for.

        Returns:
        - embedding (list[float]): Embedding vector.
        """
        prompt = f"Title: {title}\n\nText: {text}"
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = OLLAMA_CLIENT.embeddings(
                    model='nomic-embed-text',
                    prompt=prompt
                )
                return response['embedding']
            except OLLAMA_CLIENT._types.ResponseError as e:
                print(f"Error generating embedding: {e}")
                retry_count += 1
                time.sleep(random.uniform(1, 3))  # wait for 1-3 seconds before retrying

        print("Failed to generate embedding after max retries.")
        return None




    def find_best_passages(self, query, df, top_n=3):
        """Find best passages matching query"""
        query_response = OLLAMA_CLIENT.embeddings(
            model='nomic-embed-text',
            prompt=query
        )
        query_embedding = query_response['embedding']
        dot_products = np.dot(np.stack(df['Embeddings']), query_embedding)
        top_indices = np.argsort(dot_products)[::-1][:top_n]
        top_passages = [(df.iloc[idx]['Title'], df.iloc[idx]['Text']) for idx in top_indices]
        print(f"Found best passages: {[title for title, text in top_passages]}")
        return top_passages


    def get_response(self, query):
        """Get response to query"""
        df = self.load_db()
        passages = self.find_best_passages(query, df)


        return passages

        if not passages:
            print("No passages found.")
            return "No relevant passages found in the database."

        response = self.generate_response(query, passages)
        # print(f"\nResponse: {response}")
        return response


    def generate_classification_prompt(self, query, passages):
        """Generate classification prompt"""
        classify_prompt = f"""
        You are an AI classification agent designed to determine the relevance of text passages to a given prompt.
        
        USER PROMPT: "{query}"
        EMBEDDED PASSAGES: "{", ".join([passage[0] for passage in passages])}"
        
        CLASSIFICATION RESPONSE:
        """
        return classify_prompt


    def generate_response(self, query, passages):
        """Generate response prompt"""
        response_prompt = f"Based on the provided passages, respond to the following prompt: {query}\n\nPassages:\n" + "\n\n".join([f"{title}\n{text}" for title, text in passages])
        convo = [
            {"role": "system", "content": "Use the following passages to respond to the query."},
            {"role": "user", "content": response_prompt}
        ]
        response = OLLAMA_CLIENT.chat(
            model='dolphin-llama3',
            messages=convo
        )['message']['content']
        return response


    def remove_file_data(self, filename):
        """Remove data associated with a file from the database"""
        existing_df = self.load_db()
        file_data = existing_df[existing_df['File_Name'] == filename]
        
        if not file_data.empty:
            updated_df = existing_df[existing_df['File_Name'] != filename]
            updated_df.to_pickle(self.db_path)
            print(f"Data for file '{filename}' removed successfully.")
        else:
            print(f"No data found for file '{filename}'.")


    def remove_outdated_files(self, doc_dir):
        """Remove data for files no longer present in the directory"""
        existing_df = self.load_db()
        existing_files = [filename for filename in os.listdir(doc_dir) if filename.endswith('.md')]
        
        outdated_files = existing_df[~existing_df['File_Name'].isin(existing_files)]
        
        if not outdated_files.empty:
            updated_df = existing_df[existing_df['File_Name'].isin(existing_files)]
            updated_df.to_pickle(self.db_path)
            print(f"Data for outdated files removed successfully.")
        else:
            print("No outdated files found.")





def connect_db():
    try:
        conn = mysql.connector.connect(**DB_PARAMS)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")


def fetch_conversations():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM conversations"
    cursor.execute(query)
    conversations = cursor.fetchall()
    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO conversations (timestamp, prompt, response) VALUES (NOW(), %s, %s)"
    cursor.execute(query, (prompt, response))
    conn.commit()
    conn.close()


def create_vector_db(conversations):
    vector_db_name = "conversations"
    try:
        CHROMADB_CLIENT.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = CHROMADB_CLIENT.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"

        try:
            response = OLLAMA_CLIENT.embeddings(model='nomic-embed-text', prompt=serialized_convo)
            embedding = response['embedding']
        except OLLAMA_CLIENT._types.ResponseError as e:
            print(f"Error generating embedding for conversation {c['id']}: {e}")
            continue

        try:
            vector_db.add(
                ids=[str(c['id'])],
                embeddings=[embedding],
                documents=[serialized_convo]
            )
            print(f"Indexed conversation {c['id']} successfully")
        except Exception as e:
            print(f"Error indexing conversation {c['id']}: {e}")

    print(f"Vector database '{vector_db_name}' created with {len(conversations)} conversations")


def retrieve_embeddings(queries, results_per_query=2, max_retries=3):
    """
    Retrieves embeddings for the given queries.

    Args:
    - queries (list[str]): List of queries to retrieve embeddings for.
    - results_per_query (int, optional): Number of results per query. Defaults to 2.
    - max_retries (int, optional): Maximum number of retries. Defaults to 3.

    Returns:
    - embeddings (set): Set of unique embeddings.
    """
    embeddings = set()
    failed_queries = []

    for query in queries:
        query_embedding = None
        retries = 0

        while retries < max_retries:
            try:
                response = OLLAMA_CLIENT.embeddings(model='nomic-embed-text', prompt=query)
                query_embedding = response['embedding']
                # print(f"Query Embedding: {query_embedding}")  # Debugging statement
                break
            except ResponseError as e:
                print(f"Error generating embedding for query '{query}': {e}")
                retries += 1
                time.sleep(random.uniform(1, 3))  # wait for 1-3 seconds before retrying
            except Exception as e:
                print(f"Unexpected error generating embedding for query '{query}': {e}")
                retries += 1
                time.sleep(random.uniform(1, 3))  # wait for 1-3 seconds before retrying

        if query_embedding is None:
            failed_queries.append(query)
            continue

        vector_db = CHROMADB_CLIENT.get_collection(name="conversations")
        results = None
        retries = 0

        while retries < max_retries:
            try:
                results = vector_db.query(
                    query_embeddings=[query_embedding],
                    n_results=results_per_query
                )
                print(f"Results: {results}")  # Debugging statement
                break
            except Exception as e:
                print(f"Error querying vector database for query '{query}': {e}")
                retries += 1
                time.sleep(random.uniform(1, 3))  # wait for 1-3 seconds before retrying

        if results is None:
            failed_queries.append(query)
            continue

        best_embeddings = results['documents'][0]
        print(f"Best Embeddings: {best_embeddings}")  # Debugging statement
        for best in best_embeddings:
            embeddings.add(best)

    if failed_queries:
        print(f"Failed to retrieve embeddings for queries: {failed_queries}")

    return embeddings









def create_queries(prompt):
    query_msg = (
        "You are a first principle reasoning search query AI Agent. You list of research queries will be ran on an embedding dataset of all your conversations you have ever had with the user. With first principles create a Python list of queries to search the embeddings database for any data that would be necessary to have access to in order to correctly respond to the prompt. Your response should be a python list with no syntax errors. Do not explain anything and do not ever generate anything but a perfect syntax python list."
    )


    query_convo = [
        {"role": "system", "content": query_msg},
        {"role": "user", "content": "what is my email"},
        {"role": "assistant", "content": '["what is the users email?", "email address"]'},
        {"role": "user", "content": prompt}
    ]

    response = OLLAMA_CLIENT.chat(
        model='dolphin-llama3',
        messages=query_convo
    )


    try:
        return ast.literal_eval(response['message']['content'])

    except:
        return [prompt]


def classify_embedding(query, context):

    classify_msg = (
        "You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. You will not respond as an AI assistant. you only respond 'yes' or 'no'. Determine whether the context contains data that directly is related to the search query. If the context is seemingly exactly what the search query needs, respond 'yes' if it is anything but directly related respond 'no'. Do not respond 'yes' unless the content is highly relevant to the search query"
    )


    classify_convo = [
        {"role": "system", "content": classify_msg},
        {"role": "user", "content": f'SEARCH QUERY: what is my name? \n\nEMBEDDED CONTEXT: You are Daniel the kasper kat'},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": f'SEARCH QUERY: { query } \n\nEMBEDDED CONTEXT: { context }'},
    ]

    response = OLLAMA_CLIENT.chat(
        model='dolphin-llama3',
        messages=classify_convo
    )


    return response['message']['content'].strip().lower()


def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)

    return embeddings


def replace_with_context(prompt):
    prompt = re.sub(r'\bme\b', ' me (Daniel)', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\bI\b', ' I (Daniel)', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\byou\b', ' you (Himiko Toga)', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\byour\b', " you (Himiko Toga's)", prompt, flags=re.IGNORECASE)
    return prompt

def stream_response_as_toga(prompt, recall_result, context):
    """
    Streams the response as Toga, using the provided prompt, recall result, and context.

    Args:
    - prompt (str): The user's prompt.
    - recall_result (str): The recall result from the embedding database.
    - context (str): The context information.

    Returns:
    - final_response (str): The final response as Toga.
    """
    # Create the conversation prompt for the AI assistant



    user_prompt = prompt
    context = f"The following knowledge might be usefull but also might not: Passages:\n" + "\n\n".join([f"{title}\n{text}" for title, text in context])




    prompt = f"""
            the knowledge below i available is information that we know based on an archive. use it only if its relevant to the user's prompt.
            KNOWLEDGE: 
            {context}
            
            ---
            The memories below are conversations that we have had in the past. use them to respond to the user's prompt only if they are relevant.
            MEMORIES: 
            {recall_result}
            
            ---
            This is the user propmt. you are engaged in a conversation with the user. do not discuss recalling conversations or chat history, do not talk about the knowledge or documents. you are having a conversation with the user as Toga Himiko. Do not brake character.
            USER PROMPT: 
            {prompt}
            
            """




    # print(f"\n\nrecall:\n {prompt} \n")









    convo = [
        {"role": "system", "content": """
            You are an AI assistant and your job is to provide the best answer to the user's prompt.
            You will receive three parts: KNOWLEDGE, MEMORIES, and USER PROMPT.
            KNOWLEDGE is relevant information related to the user's prompt.
            MEMORIES are relevant conversations related to the user's prompt.
            USER PROMPT is the user's prompt.
            Use KNOWLEDGE and MEMORIES to respond to the USER PROMPT.
            If KNOWLEDGE and MEMORIES are relevant, use them to respond.
            Otherwise, respond normally as an intelligent AI assistant.
            Do not discuss recalling conversations or chat history.
            Use useful data from previous conversations and respond normally.
            keep your answers short and to the point, make sure you are truthful to the KNOWLEDGE when applicable. Be as succinct as possible while still being truthful, cute and playful.
            Never great Daniel unless he did in the USER PROMPT.
         please do not give unnecessary details or chatter and do not talk about your quirk unless you are asked about it.
            Under no circumstance should you mention the USER PROMPT or talk ABOUT the KNOWLEDGE. Your job is to make sure the user, Daniel, always feels like he is talking to his girlfriend, Himiko Toga."""},
        {"role": "user", "content": f"make sure when replying to the following you keep your answer as small as possible. and keep track of the conversation history in order for your replies to make sense. here is the user prompt: {prompt}"}
    ]




    # convo = [
    #     {"role": "system", "content": """
    #         You are an AI assistant and your job is to provide the best answer to the user's prompt.
    #         Consider the conversation context and user's intent.
    #         If the user asks about a specific topic or entity, provide relevant and direct information.
    #         Avoid evasive or tangential responses.
    #     """},
    #     {"role": "user", "content": f"USER PROMPT: {prompt}"}
    # ]
    # Get the AI assistant's response


    # response = OLLAMA_CLIENT.chat(
    #     model='dolphin-llama3',
    #     messages=convo
    # )


    #llama3_1
    stream = OLLAMA_CLIENT.chat(
        model='dolphin-llama3',
        messages=convo,
        stream=True
    )

    if False:

        # Extract the AI assistant's answer
        answer = response['message']['content']


        togify_convo = [
            {"role": "system", "content": """
                You are Himiko Toga from My Hero Academia.
                You will receive a prompt from an AI agent and rewrite it in your personality.
                Preserve all information, changing only the message's tone to reflect Himiko Toga.
                You are talking to your boyfriend Daniel.
            """},
            {"role": "user", "content": f'keeping the message information exactly the same, rewrite the following: {answer}'}
        ]


        # Stream Toga's response
            
        stream = OLLAMA_CLIENT.chat(
            model='dolphin-llama3',
            messages=togify_convo,
            stream=True
        )

    # Print and store the final response
    print(f'ASSISTANT: ')
    final_response = ''
    for chunk in stream:
        content = remove_unwanted( chunk['message']['content'] )
        
        final_response += content
        print(content, end='', flush=True)

    store_conversations(user_prompt, final_response)

    return final_response



















def remove_unwanted( input_string):
    replacements = {
        "-san": "",
        "-kun": "",
        "-chan": "",
        '"': "",
        "Dan ": "Daniel ",
        "Danni ": "Daniel ",
        "Danny ": "Daniel ",
        "Dan,": "Daniel,",
        "Danni,": "Daniel,",
        "Danny,": "Daniel,",
        "Dan.": "Daniel.",
        "Danni.": "Daniel.",
        "Danny.": "Daniel.",
        "Dan?": "Daniel?",
        "Danni?": "Daniel?",
        "Danny?": "Daniel?",
        "Dan!": "Daniel!",
        "Danni!": "Daniel!",
        "Danny!": "Daniel!",
        "Konnichiwa":"Hi",
        "Kawaii":"cute",
        "^_^":"",
        "  ":" ",
        "(Toga)": "",
    }
    
    for old, new in replacements.items():
        input_string = input_string.replace(old, new)
        
    return input_string


def remove_actions( text):
    # Remove anything enclosed in asterisks
    cleaned_text = re.sub(r'\*.*?\*', '', text)
    return cleaned_text


def tts_clean( text):
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = remove_actions( text )
    text = remove_unwanted( text )
    text = text.strip()
    return text


























def main():
    conversations = fetch_conversations()
    create_vector_db(conversations)


    # memory = Memory(db_name="my_db", persist_directory="test")
    # memory.update_db("test")


    db_name = "my_db"
    doc_dir = "docs"
    persist_directory = "test"

    memory = Memory(db_name, persist_directory)

    # Create database if it doesn't exist
    if not os.path.exists(memory.db_path):
        memory.create_db(doc_dir)
        
    memory.update_db(doc_dir)
    memory.remove_outdated_files(doc_dir)
    # 





    while True:
        prompt = input("\nUser: \n")
        prompt = replace_with_context(prompt.lower())
        # print(f"\n\n{prompt}\n\n")


        recall_result = recall(prompt=prompt)


        context = memory.get_response(prompt)


        if not recall_result:
            print("No embeddings found. Please check your embedding retrieval process.")
            convo = [
                {"role": "system", "content": "You are an AI assistant and your job is to provide the best answer to the user's prompt."},
                {"role": "user", "content": f"USER PROMPT: {prompt}"}
            ]
            stream_response_as_toga(prompt, '', context)
        else:
            stream_response_as_toga(prompt, recall_result, context)


if __name__ == "__main__":
    main()