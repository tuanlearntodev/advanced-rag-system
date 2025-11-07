from dotenv import load_dotenv
from graph.graph import app
load_dotenv()

if __name__ == "__main__":
    print("Starting the graph application...")
    app.invoke({"question": "Why was Socrates executed?", "crag": True})
