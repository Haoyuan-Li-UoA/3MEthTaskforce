import argparse
from original_data_treatment import original_data_treatment

# 定义函数来解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Origin Data Preprocessing Parameter Config")

    # 添加命令行参数
    parser.add_argument('--token_num', type=int, default=200, help="number of token")
    parser.add_argument('--sparse', action='store_true', help="sample sparse data")
    parser.add_argument('--random_sample', action='store_true', help="random sample")
    parser.add_argument('--dense', action='store_true', help="sample dense data")
    parser.add_argument('--token_list', type=str, nargs='+', default=None, help="use specific token choices")
    parser.add_argument('--each_token', type=int, default=500, help="number of recording for each token")

    # 解析命令行参数
    return parser.parse_args()

# 主函数
def main():
    # 获取命令行参数
    args = parse_args()

    # 调用 original_data_treatment 函数并传入参数
    original_data_treatment(
        token_num=args.token_num,
        sparse=args.sparse,
        random_sample=args.random_sample,
        dense=args.dense,
        token_list=args.token_list,
        each_token=args.each_token
    )

if __name__ == "__main__":
    main()
