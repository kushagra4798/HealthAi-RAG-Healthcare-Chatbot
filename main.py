from chatbot_core import build_index, ask_chatbot, get_greeting, BOT_NAME


def main():
    print(BOT_NAME + " is loading the knowledge base, please wait...")
    index = build_index()
    print()
    print(get_greeting())
    print("(Type 'exit' to quit)\n")

    history = []
    followup_asked = False

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Take care of your health. Goodbye!")
            break

        answer, score, specialty, followup_asked = ask_chatbot(query, index, history, followup_asked)

        print(BOT_NAME + ":", answer)
        if score is not None:
            confidence_percent = round(score * 100, 1)
            print("(matched specialty: " + str(specialty) + ", confidence: " + str(confidence_percent) + "%)")
        print()

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
