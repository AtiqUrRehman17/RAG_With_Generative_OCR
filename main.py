import os
from dotenv import load_dotenv
from chatbot.core import ScannedPDFChatbot

load_dotenv()


def main():
    print("=== Scanned PDF RAG Chatbot (STABLE BUILD) ===\n")

    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ File not found.")
        return

    bot = ScannedPDFChatbot(pdf_path)
    bot.initialize()

    print("\nType 'exit' to quit\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print(f"\nAssistant: {bot.ask(q)}\n")


if __name__ == "__main__":
    main()
