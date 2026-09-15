from src.services.research_service import run_research


def main():
    query = input("Enter your research topic: ")

    result = run_research(query)

    print("\n" + "=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()