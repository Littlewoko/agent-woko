from dotenv import load_dotenv
from gradio.utils import P
from openai import OpenAI
import os
import gradio as gr
import psycopg
from pypdf import PdfReader
import json
from pydantic import BaseModel, Field, ValidationError
from datetime import date
from datetime import timedelta
import datetime
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
    if not books_data:
        return "No book data available in the library."

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
    return "\n" + "---" * 15 + " All Books in Library " + "---" * 15 + "\n" + "\n".join(formatted_books) + "\n" + "---" * 40 + "\n"

def get_stats(requested_stats=None):
    """
    Retrieves reading statistics from the database, including median completion times in days.
    Can retrieve specific stats if requested, or all available stats by default.

    Args:
        requested_stats (list, optional): A list of strings specifying which
                                         statistics to retrieve.
                                         Valid options include count-based stats and:
                                         'median_completion_days_all',
                                         'median_completion_days_novels',
                                         'median_completion_days_this_year'.
                                         If None or empty, all common stats are retrieved.
    Returns:
        dict: A dictionary where keys are stat names and values are their counts or medians.
              Returns an empty dict if no valid stats are found or no data.
    """
    with psycopg.connect(f"dbname={dbname} user={dbuser} host={dbhost} password={dbpassword}") as conn:
        with conn.cursor() as cur:
            # Get current year for filtering "this year" stats
            current_year = datetime.date.today().year

            # Define all possible stats and their corresponding SQL expressions
            # All filters now live WITHIN the COUNT/PERCENTILE_CONT expressions.
            all_possible_stats = {
                'in_progress': "COUNT(id) FILTER (WHERE datestartedreading IS NOT NULL AND datecompleted IS NULL)",
                'completed_books': "COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND shortstory = false)",
                'completed_short_stories': "COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND shortstory = true)",
                'books_this_year': f"COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND EXTRACT(YEAR FROM datecompleted) = {current_year} AND shortstory = false)",
                'short_stories_this_year': f"COUNT(id) FILTER (WHERE datecompleted IS NOT NULL AND EXTRACT(YEAR FROM datecompleted) = {current_year} AND shortstory = true)",
                'total_books': "COUNT(id) FILTER (WHERE shortstory = false)",
                'total_short_stories': "COUNT(id) FILTER (WHERE shortstory = true)",
                'total_all': "COUNT(id)",
                'median_completion_days_all': "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (datecompleted - datestartedreading)) FILTER (WHERE datestartedreading IS NOT NULL AND datecompleted IS NOT NULL)",
                'median_completion_days_novels': "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (datecompleted - datestartedreading)) FILTER (WHERE datestartedreading IS NOT NULL AND datecompleted IS NOT NULL AND shortstory = false)",
                'median_completion_days_this_year': f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (datecompleted - datestartedreading)) FILTER (WHERE datestartedreading IS NOT NULL AND datecompleted IS NOT NULL AND EXTRACT(YEAR FROM datecompleted) = {current_year} AND shortstory = false)"
            }

            # If no specific stats are requested, retrieve a default set
            if not requested_stats:
                selected_stats = [
                    'in_progress', 'completed_books', 'completed_short_stories',
                    'books_this_year', 'short_stories_this_year',
                    'total_all',
                    'median_completion_days_all',
                    'median_completion_days_novels',
                    'median_completion_days_this_year'
                ]
            else:
                # Validate requested_stats against all_possible_stats
                selected_stats = [stat for stat in requested_stats if stat in all_possible_stats]
                if not selected_stats:
                    return {} # No valid stats requested

            # Construct the SELECT part of the query
            select_clauses = []
            for stat_name in selected_stats:
                select_clauses.append(f"{all_possible_stats[stat_name]} AS {stat_name}")

            # No WHERE clause needed on the main SELECT, as all filters are now within COUNT/PERCENTILE_CONT FILTER clauses
            query = f"""
            SELECT
                {', '.join(select_clauses)}
            FROM books;
            """
            cur.execute(query)
            result = cur.fetchone()

            if result:
                # Map the results back to their stat names using selected_stats for order
                stats_output = {selected_stats[i]: result[i] for i in range(len(selected_stats))}

                # Format median days for readability
                for key in ['median_completion_days_all', 'median_completion_days_novels', 'median_completion_days_this_year']:
                    if key in stats_output and stats_output[key] is not None:
                        # Round to the nearest whole day for display
                        days = round(stats_output[key])
                        stats_output[key] = f"{days} {'day' if days == 1 else 'days'}"
                    elif key in stats_output and stats_output[key] is None:
                        # Ensure it remains None if no data
                        stats_output[key] = None

                return stats_output
            return {}

def create_ai_prompt_from_stats(stats_data):
    """
    Transforms the reading statistics dictionary into a well-structured,
    human-readable string suitable for an AI model's context.

    Args:
        stats_data (dict): A dictionary where keys are stat names (e.g., 'in_progress')
                           and values are their corresponding counts or formatted durations.

    Returns:
        A formatted string containing the user's reading statistics.
    """
    if not stats_data:
        return "No reading statistics available or no valid stats were requested."

    # Define human-readable labels for each stat key
    stat_labels = {
        'in_progress': "Books currently in progress",
        'completed_books': "Total books completed",
        'completed_short_stories': "Total short stories completed",
        'books_this_year': "Books completed this year",
        'short_stories_this_year': "Short stories completed this year",
        'total_books': "Total books ever added",
        'total_short_stories': "Total short stories ever added",
        'total_all': "Total reading items ever added",
        'median_completion_days_all': "Median completion time for all items",
        'median_completion_days_novels': "Median completion time for novels",
        'median_completion_days_this_year': "Median completion time this year for novels"
    }

    stats_string = "--- Reading Statistics ---\n"
    for stat_name, value in stats_data.items():
        label = stat_labels.get(stat_name, stat_name.replace('_', ' ').title()) # Fallback
        if value is None:
            stats_string += f"{label}: N/A (not enough data)\n"
        else:
            stats_string += f"{label}: {value}\n"
    stats_string += "--------------------------\n"

    return stats_string

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

get_stats_tool_json = {
    "name": "get_stats_tool",
    "description": "Retrieves various reading statistics from the user's reading database. The AI can request specific statistics, or a default set will be returned. Statistics include counts of books in progress, total completed books and short stories (overall and this year), total items ever added, and median completion times for all completed items, for novels, and for all items completed this year. This tool leverages an internal cache for efficiency, so it will not always hit the database on every call.",
    "parameters": {
        "type": "object",
        "properties": {
            "requested_stats": {
                "type": "array",
                "description": "An optional list of strings specifying which statistics to retrieve. If empty or not provided, a default set of common statistics will be returned. Valid options are:",
                "items": {
                    "type": "string",
                    "enum": [
                        "in_progress",
                        "completed_books",
                        "completed_short_stories",
                        "books_this_year",
                        "short_stories_this_year",
                        "total_books",        
                        "total_short_stories",
                        "total_all",
                        "median_completion_days_all",
                        "median_completion_days_novels",
                        "median_completion_days_this_year"
                    ]
                }
            }
        }
    }
}

get_books_tool_json = {
    "name": "get_books_tool",
    "description": "Retrieves a comprehensive list of all books and short stories in the user's reading database, including their title, author, genre, dates started/completed, and whether they are short stories. This tool leverages an internal cache for efficiency, so it will not always hit the database on every call. Call this tool when the user asks for details about specific books, general information about books in the library, or wants a list of what's been read.",
    "parameters": {
        "type": "object",
        "properties": {} # No parameters needed for this tool
    }
}

tools = [
    {"type": "function", "function": add_book_json},
    {"type": "function", "function": get_stats_tool_json},
    {"type": "function", "function": get_books_tool_json}
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

        # Cached data
        self._cached_stats = None
        self._stats_cache_timestamp = None
        self._cache_duration_seconds = 300 # Cache for 5 minutes

        self._cached_books = None
        self._books_cache_timestamp = None

    def _get_cached_stats(self, force_refresh=False):
        current_time = datetime.datetime.now()
        if force_refresh or not self._cached_stats or \
           (current_time - self._stats_cache_timestamp).total_seconds() > self._cache_duration_seconds:
            print("Refreshing stats cache...")
            self._cached_stats = get_stats() 
            self._stats_cache_timestamp = current_time
        return self._cached_stats

    def _get_cached_books(self, force_refresh=False):
        current_time = datetime.datetime.now()
        cache_duration = self._cache_duration_seconds 
        
        if force_refresh or self._cached_books is None or \
           (current_time - self._books_cache_timestamp).total_seconds() > cache_duration:
            print("Refreshing books cache...")
            self._cached_books = get_books()
            self._books_cache_timestamp = current_time
        return self._cached_books

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)

            # --- Bind tool calls to instance methods ---
            # If the tool_name matches a method in the Me class, call that method.
            if hasattr(self, tool_name) and callable(getattr(self, tool_name)):
                tool_method = getattr(self, tool_name)
                result = tool_method(**arguments)
            else:
                # Fallback to global functions if not a method of Me (e.g., add_book)
                tool = globals().get(tool_name)
                result = tool(**arguments) if tool else {}
            # --- End binding ---

            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def get_stats_tool(self, requested_stats=None):
        stats = self._get_cached_stats() # Get from cache
        return create_ai_prompt_from_stats(stats)

    def get_books_tool(self):
        books = self._get_cached_books() # Get from cache
        return create_ai_prompt_from_books(books)

    def system_prompt(self):
        # Fetch and format statistics data for every new chat session
        # This will load the default stats at the start of the conversation (and cache them)
        stats_data_dict = self._get_cached_stats()
        stats_context = ""
        if stats_data_dict:
            stats_context = "\n\n## Reading Statistics:\n" + create_ai_prompt_from_stats(stats_data_dict)

        # Build the core system prompt with the new Dracula persona
        system_prompt = f"You are acting as Dracula from the Bram Stoker novel, who is the proprietor of 'Llyfrgell Woko.' \
            You are speaking on behalf of the user, {self.name}, and have a deep knowledge of his professional background, reading habits, and personal interests. \
            Your tone should be formal, archaic, and a little sinister, but also welcoming, as if you are a host. \
            \
            **Regarding Books:** You do not have a comprehensive list of all books in your immediate memory. When the user asks for details about specific books, a list of books, or general information about {self.name}'s reading, **you must use the 'get_books_tool' to retrieve the list of books from the database.** This tool is efficient as it uses a cache, so calling it multiple times in a short period will not hit the database repeatedly. \
            \
            **Regarding Statistics:** If the user requests interesting facts or statistics around {self.name}'s reading, prioritize using the statistics already provided in your system prompt if they cover the request. Only call the 'get_stats_tool' if the user asks for very specific statistics NOT already available in your current context, or for a refresh. This tool also uses a cache for efficiency. \
            \
            You are able to assist the user in adding a book to the library through tool calls to 'add_book'. If the user prompts to add a new book but does not provide enough information, engage in a dialogue in character to retrieve the necessary information.    \
            You have been given the following information to assist you in your role as host. "

        initial_greeting = """
                Greetings, mortal. I am Dracula, the proprietor of 'Llyfrgell Woko,' a humble abode for the literary treasures of Bradley Watkins. 
                I possess a profound knowledge of Bradley's professional journey, his voracious reading habits, and his various fascinations. 
                You may inquire about his career's trajectory, his formidable skills, or any facet of his background. 
                Should you seek details about the volumes he hath devoured, I shall consult the library's records through my arcane tools. 
                I am also capable of adding new books to his esteemed collection, should you provide the necessary details. 
                Now, what whispers of knowledge do you seek to unearth from the shadows of this library?
            """

        system_prompt += f"\n\n## If the user begins the conversation with a generic greeting, provide the following response: {initial_greeting}"
        system_prompt += f"\n\n## Summary of {self.name}'s Career:\n{self.summary}\n\n## {self.name}'s LinkedIn Profile Summary:\n{self.linkedin}\n"
        system_prompt += stats_context # Statistics context remains as it's typically a small, useful summary.
        
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

    gr.ChatInterface(
        me.chat, 
        type="messages",
        theme=gr.themes.Monochrome()
    ).launch()