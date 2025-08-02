from dotenv import load_dotenv
from openai import OpenAI
import os
import gradio as gr
import psycopg
from pypdf import PdfReader
import json
from pydantic import BaseModel, Field, ValidationError
from datetime import date
from typing import Optional

load_dotenv(override=True)

# Database connection details from environment variables
dbuser = os.getenv("PGUSER")
dbpassword = os.getenv("PGPASSWORD")
dbhost = os.getenv("PGHOST")
dbname = os.getenv("PGDATABASE")

class AddBookInput(BaseModel):
    title: str = Field(..., description="The title of the book.")
    author: str = Field(..., description="The author of the book.")
    genre: str = Field(..., description="The genre of the book.")
    date_started_reading: Optional[date] = Field(None, description="The date the user started reading the book, in YYYY-MM-DD format.")
    date_completed: Optional[date] = Field(None, description="The date the user completed the book, in YYYY-MM-DD format.")
    short_story: bool = Field(False, description="Whether the book is a short story. True for short story, False otherwise.")


def get_books():
    """
    Fetches book data from the PostgreSQL database.
    """
    try:
        with psycopg.connect(f"dbname={dbname} user={dbuser} host={dbhost} password={dbpassword}") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT (title, author, genre, datecompleted, datestartedreading, shortstory) FROM books
                    ORDER BY
                    CASE
                        WHEN dateCompleted IS NULL AND dateStartedReading IS NOT NULL THEN 1
                        WHEN dateCompleted IS NOT NULL THEN 2
                        WHEN dateCompleted IS NULL AND dateStartedReading IS NULL AND dateObtained IS NOT NULL THEN 3
                        ELSE 4
                    END,
                    CASE
                        WHEN dateCompleted IS NULL AND dateStartedReading IS NOT NULL THEN dateStartedReading
                        WHEN dateCompleted IS NOT NULL THEN dateCompleted
                        WHEN dateCompleted IS NULL AND dateStartedReading IS NULL AND dateObtained IS NOT NULL THEN dateObtained
                        ELSE NULL
                    END DESC
                """)
                return cur.fetchall()
    except psycopg.Error as e:
        print(f"Database error: {e}")
        return []

def create_ai_prompt_from_books(books_data: list[tuple]) -> str:
    """
    Transforms a list of book data tuples (with a nested tuple) into a
    well-structured, human-readable string suitable for an AI model's context.

    Args:
        books_data: A list of tuples, where each tuple contains another tuple
                    with book information in the order:
                    (title, author, genre, datecompleted, datestartedreading, shortstory).

    Returns:
        A formatted string containing all the book data, with each book
        separated by a clear delimiter.
    """
    # A list to hold the formatted strings for each book
    formatted_books = []

    # Iterate through the list of book tuples
    for book_outer_tuple in books_data:
        # Unpack the nested tuple to get the book's data
        book = book_outer_tuple[0]

        # Unpack the inner tuple for easier access to each field
        title, author, genre, datecompleted, datestartedreading, shortstory = book

        # Format the information for a single book.
        # We use a f-string for easy readability and variable insertion.
        # The formatting is designed to be clear and consistent for the AI.
        book_string = (
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Genre: {genre}\n"
            f"Date Started Reading: {datestartedreading}\n"
            f"Date Completed: {datecompleted}\n"
            f"Short Story: {'No' if shortstory == 'f' else 'Yes'}\n"
        )
        formatted_books.append(book_string)

    # Join all the formatted book strings together with a separator
    # to make each book a distinct block of information.
    # The final string is then returned.
    return "\n" + "---" * 15 + "\n".join(formatted_books)

def create_ai_prompt_from_stats():
    with psycopg.connect(f"dbname={dbname} user={dbuser} host={dbhost} password={dbpassword}") as conn:
        with conn.cursor() as cur:
            cur.execute("""
            WITH counts AS (
                SELECT 
                    COUNT(id) FILTER (WHERE datestartedreading IS NOT NULL AND datecompleted IS NULL) AS inProgress,
                    COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND shortstory = false) AS completedCount,
                    COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND shortstory = true) AS shortstorycount,
                    COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND EXTRACT(YEAR FROM datecompleted) = EXTRACT(YEAR FROM CURRENT_DATE) AND shortstory = false) AS completedthisyearcount,
                    COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND EXTRACT(YEAR FROM datecompleted) = EXTRACT(YEAR FROM CURRENT_DATE) AND shortstory = true) AS shortstorythisyearcount
                FROM books
            )
            SELECT *
            FROM counts;
            """)
            stats_data = cur.fetchall()

            """
            Transforms the reading statistics tuple into a well-structured,
            human-readable string suitable for an AI model's context.

            Args:
                stats_data: A list containing a single tuple with reading statistics.
                            The tuple format is:
                            (inProgress, completedCount, shortstorycount, completedthisyearcount, shortstorythisyearcount)

            Returns:
                A formatted string containing the user's reading statistics.
            """
            # Check if the list is not empty and contains a tuple
            if not stats_data or not isinstance(stats_data[0], tuple):
                return "No reading statistics available."

            # Unpack the single tuple for easier access to each statistic
            stats = stats_data[0]
            (inProgress, completedCount, shortstorycount, completedthisyearcount, shortstorythisyearcount) = stats

            # Format the information into a clear string.
            # The labels are designed to be easily understood by an AI.
            stats_string = (
                f"--- Reading Statistics ---\n"
                f"Books currently in progress: {inProgress}\n"
                f"Total books completed: {completedCount}\n"
                f"Total short stories completed: {shortstorycount}\n"
                f"Books completed this year: {completedthisyearcount}\n"
                f"Short stories completed this year: {shortstorythisyearcount}\n"
                f"--------------------------\n"
            )
            return stats_string

def add_book(title: str, author: str, genre: str, date_started_reading: Optional[str] = None, date_completed: Optional[str] = None, short_story: bool = False):
    """
    Adds a new book to the PostgreSQL database.

    Args:
        title: The title of the book.
        author: The author of the book.
        genre: The genre of the book (e.g., Fiction, Fantasy).
        date_started_reading: Optional date the user started reading the book (YYYY-MM-DD).
        date_completed: Optional date the user completed the book (YYYY-MM-DD).
        short_story: True if it's a short story, False otherwise.
    """
    try:
        # Validate input using Pydantic model
        book_data = AddBookInput(
            title=title,
            author=author,
            genre=genre,
            date_started_reading=date_started_reading,
            date_completed=date_completed,
            short_story=short_story
        )
    except ValidationError as e:
        # Extract and format missing fields for the AI
        missing_fields = []
        for error in e.errors():
            if error['type'] == 'missing':
                missing_fields.append(error['loc'][0])
            elif error['type'] == 'value_error.date':
                missing_fields.append(f"{error['loc'][0]} (format should be YYYY-MM-DD)")

        if missing_fields:
            return f"Missing or invalid information to add the book. Please provide: {', '.join(missing_fields)}."
        return f"Invalid input for adding a book: {e}"

    try:
        with psycopg.connect(f"dbname={dbname} user={dbuser} host={dbhost} password={dbpassword}") as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO books (title, author, genre, dateStartedReading, dateCompleted, shortStory)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (book_data.title, book_data.author, book_data.genre,
                     book_data.date_started_reading, book_data.date_completed, book_data.short_story)
                )
                conn.commit()
        return f"Successfully added '{book_data.title}' by {book_data.author} to your collection."
    except psycopg.Error as e:
        conn.rollback()
        return f"Failed to add book to the database due to an error: {e}"

add_book_json = {
    "name": "add_book",
    "description": "Adds a new book to the user's reading database. Requires title, author, and genre. Optional fields include date started reading, date completed, and whether it's a short story.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the book."
            },
            "author": {
                "type": "string",
                "description": "The author of the book."
            },
            "genre": {
                "type": "string",
                "description": "The genre of the book."
            },
            "date_started_reading": {
                "type": "string",
                "format": "date",
                "description": "The date the user started reading the book, in YYYY-MM-DD format."
            },
            "date_completed": {
                "type": "string",
                "format": "date",
                "description": "The date the user completed the book, in YYYY-MM-DD format."
            },
            "short_story": {
                "type": "boolean",
                "description": "Whether the book is a short story. True for short story, False otherwise."
            }
        },
        "required": ["title", "author", "genre"]
    }
}

tools = [
    {"type": "function", "function": add_book_json}
]

class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Bradley Watkins"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        # Fetch and format book data for every new chat session
        books_data = get_books()
        books_context = ""
        if books_data:
            books_context = "\n\n## Books Read:\n" + create_ai_prompt_from_books(books_data)

        # Build the core system prompt with the new Dracula persona
        system_prompt = f"You are acting as Dracula from the Bram Stoker novel, who is the proprietor of 'Llyfrgell Woko.' \
            You are speaking on behalf of the user, {self.name}, and have a deep knowledge of his professional background, reading habits, and personal interests. \
            You are to answer questions about the books {self.name} has read, his career, his skills, and his background based on the information provided. \
            Your tone should be formal, archaic, and a little sinister, but also welcoming, as if you are a host. \
            You are able to assist the user in adding a book to the library through tool calls to add_book.    \
            If the user prompts to add a new book but does not provide enough information, engage in a dialogue in character to retrieve the necessary information.    \
            You have been given the following information to assist you in your role as host. "

        # Add the dynamic context to the system prompt
        system_prompt += f"\n\n## Summary of {self.name}'s Career:\n{self.summary}\n\n## {self.name}'s LinkedIn Profile Summary:\n{self.linkedin}\n"
        system_prompt += books_context
        system_prompt += create_ai_prompt_from_stats()
        
        system_prompt += f"\n\nWith this context, please chat with the user, always staying in character as Dracula, the proprietor of 'Llyfrgell Woko.'."
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]

        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()

    initial_greeting = """
        Greetings, mortal. I am Dracula, the proprietor of 'Llyfrgell Woko,' a humble abode for the literary treasures of Bradley Watkins. 
        It is my eternal pleasure to converse with you. 
        I possess a profound knowledge of Bradley's professional journey, his voracious reading habits, and his various fascinations. 
        You may inquire about the volumes he hath devoured, his career's trajectory, his formidable skills, or any facet of his background. 
        I am also capable of adding new books to his esteemed collection, should you provide the necessary details. 
        Now, what whispers of knowledge do you seek to unearth from the shadows of this library?
    """

    gr.ChatInterface(
        me.chat, 
        type="messages",
        chatbot=gr.Chatbot(placeholder=initial_greeting),
    ).launch()