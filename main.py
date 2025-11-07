from dotenv import load_dotenv
from graph.graph import app
load_dotenv()

if __name__ == "__main__":
    print("Starting the graph application...")
    print(app.invoke({"question": "What is Mill's definition of higher pleasure?", "crag": False}))
