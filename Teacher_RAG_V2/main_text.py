from src.rag_engine import TeacherRAG


def main():
    rag = TeacherRAG()

    print("\nAI Teacher Text Mode")
    print("Type your question. Type 'quit' to exit.\n")

    while True:
        question = input("Student: ").strip()

        if question.lower() in {"quit", "exit"}:
            break

        answer, docs = rag.ask(question, show_sources=True)

        print("\nTeacher:")
        print(answer)

        print("\nSources:")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_file", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "?")
            snippet = doc.page_content[:180].replace("\n", " ")
            print(f"{i}. {source} | chunk {chunk_index} | {snippet}...")

        print("-" * 70)


if __name__ == "__main__":
    main()