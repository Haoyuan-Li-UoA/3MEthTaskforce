import os
import random

# Function 1: data_path_researcher
def data_path_researcher():
    # 获取当前文件路径
    current_path = os.getcwd()

    # 回到上上级文件夹
    base_path = os.path.abspath(os.path.join(current_path, "..", ".."))

    # 定义各个数据文件夹的路径
    token_transaction_path = os.path.join(base_path, "3MEthTaskforce Data", "Token Transaction", "Transaction 3880")
    token_info_history_path = os.path.join(base_path, "3MEthTaskforce Data", "Token Info", "Token Historical Data 3880 with return")
    token_info_general_path = os.path.join(base_path, "3MEthTaskforce Data", "Token Info", "token_general_3880.csv")
    global_data_path = os.path.join(base_path, "3MEthTaskforce Data", "Globa Data")
    textual_data_path = os.path.join(base_path, "3MEthTaskforce Data", "Reddit Textual", "Reddit_clean.csv")
    textual_save_path = os.path.join(base_path, "3MEthTaskforce Data", "Reddit Textual", "textual_clean.csv")
    textual_formula_path = os.path.join(base_path, "3MEthTaskforce Data", "Reddit Textual", "textual_formula.csv")

    # 返回所有路径
    return {
        "token_transaction_path": token_transaction_path,
        "token_info_history_path": token_info_history_path,
        "token_info_general_path": token_info_general_path,
        "global_data_path": global_data_path,
        "textual_data_path": textual_data_path,
        "textual_save_path": textual_save_path,
        "textual_formula_path":textual_formula_path
    }

# Function 2: data_combination
def data_combination(num=100, sparse=False, random_sample=False, dense=False, token_list=None, path=""):
    sample_list = []

    # 获取文件夹中的所有文件信息
    files_info = [(os.path.splitext(f)[0], os.path.getsize(os.path.join(path, f))) for f in os.listdir(path) if f.endswith('.csv')]

    # 按照文件大小进行排序
    files_info_sorted = sorted(files_info, key=lambda x: x[1])

    # 生成token address列表
    token_address_list = [file[0] for file in files_info_sorted]
    assert num <= len(token_address_list), f"num value: {num} should smaller than the length of token_list: {len(token_address_list)}"

    if sparse:
        # 从小文件开始选，直到达到num的数量
        sample_list = token_address_list[:num]
    elif random_sample:
        # 随机抽取num个文件
        sample_list = random.sample(token_address_list, num)
    elif dense:
        # 从大文件开始选
        sample_list = token_address_list[-num:]
    elif token_list:
        # 验证token_list中的每个元素是否在token_address_list中
        assert all(token in token_address_list for token in token_list), "token address error"
        sample_list = token_list

    return sample_list


TEST = False

# Testing the functions
if TEST:

    # Test data_path_researcher
    paths = data_path_researcher()
    print("Paths:", paths)

    # Test data_combination with different conditions
    # Assuming the path returned from data_path_researcher() is correct
    transaction_path = paths["token_transaction_path"]

    # Test sparse data sampling
    sparse_sample = data_combination(num=10, sparse=True, path=transaction_path)
    print("Sparse Sample:", sparse_sample)

    # Test random data sampling
    random_sample = data_combination(num=10, random_sample=True, path=transaction_path)
    print("Random Sample:", random_sample)

    # Test dense data sampling
    dense_sample = data_combination(num=10, dense=True, path=transaction_path)
    print("Dense Sample:", dense_sample)

    # Test specific token_list
    specific_token_list = ['0xf629cbd94d3791c9250152bd8dfbdf380e2a3b9c', '0xf4d2888d29d722226fafa5d9b24f9164c092421e', '0x8e870d67f660d95d5be530380d0ec0bd388289e1']
    specific_sample = data_combination(token_list=specific_token_list, path=transaction_path)
    print("Specific Token Sample:", specific_sample)