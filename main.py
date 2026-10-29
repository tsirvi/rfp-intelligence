"""Entry point: python main.py --bid ./data/Bid1  |  python main.py --ask "question" """
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP Intelligence Platform")
    parser.add_argument("--bid", help="Path to a bid folder to index and extract")
    parser.add_argument("--ask", help="Question to answer over indexed bids")
    args = parser.parse_args()
    parser.print_help() if not (args.bid or args.ask) else print("Not implemented yet")


if __name__ == "__main__":
    main()
