import argparse
from original_data_treatment import original_data_treatment


# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Origin Data Preprocessing Parameter Config")

    # Add command-line arguments
    parser.add_argument('--token_num', type=int, default=200, help="number of tokens")
    parser.add_argument('--sparse', action='store_true', help="sample sparse data")
    parser.add_argument('--random_sample', action='store_true', help="random sample")
    parser.add_argument('--dense', action='store_true', help="sample dense data")
    parser.add_argument('--token_list', type=str, default=None, choices=['test_sample'],
                        help="use specific token choices")
    parser.add_argument('--strategy', type=str, default='time_chunk_sample',
                        choices=['time_chunk_sample', 'entire_token_recording'],
                        help="number of recordings for each token")
    parser.add_argument('--from_time', type=str, default='2022-05-01 00:00:00', help='Time chunk start date')
    parser.add_argument('--to_time', type=str, default='2022-05-30 00:00:00', help='Time chunk end date')
    parser.add_argument('--task', type=str, default='link', choices=['link', 'link_and_price_prediction'],
                        help='Task type')
    parser.add_argument('--only_consider_buy', action='store_true', help="data will only consider buying behavior")

    # Parse command-line arguments
    return parser.parse_args()


# Main function
def main():
    # Get command-line arguments
    args = parse_args()

    # Call the original_data_treatment function and pass the parameters
    original_data_treatment(
        token_num=args.token_num,
        sparse=args.sparse,
        random_sample=args.random_sample,
        dense=args.dense,
        token_list=args.token_list,
        strategy=args.strategy,
        from_time=args.from_time,
        to_time=args.to_time,
        task=args.task,
        only_consider_buy=args.only_consider_buy
    )


if __name__ == "__main__":
    main()
