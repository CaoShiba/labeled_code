import pickle
from collections import Counter


# 从pickle文件中加载数据
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data


# 将数据拆分为单个查询和多个查询
def split_data(total_data, qids):
    result = Counter(qids)  # 统计每个qid出现的次数
    total_data_single = []  # 存放单个查询的数据
    total_data_multiple = []  # 存放多个查询的数据
    for data in total_data:
        if result[data[0][0]] == 1:  # 判断该qid是否只出现一次
            total_data_single.append(data)
        else:
            total_data_multiple.append(data)
    return total_data_single, total_data_multiple


# 处理staqc数据，将其拆分为单个查询和多个查询，并保存结果
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    # 读取文件内容
    with open(filepath, 'r') as f:
        total_data = eval(f.read())
    # 提取每条数据的qid
    qids = [data[0][0] for data in total_data]
    # 拆分数据
    total_data_single, total_data_multiple = split_data(total_data, qids)

    # 保存单个查询的数据
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    # 保存多个查询的数据
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


# 处理大语料数据，将其拆分为单个查询和多个查询，并保存结果
def data_large_processing(filepath, save_single_path, save_multiple_path):
    # 加载数据
    total_data = load_pickle(filepath)
    # 提取每条数据的qid
    qids = [data[0][0] for data in total_data]
    # 拆分数据
    total_data_single, total_data_multiple = split_data(total_data, qids)

    # 保存单个查询的数据
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    # 保存多个查询的数据
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


# 将未标注的单一查询数据标注为带有标签的数据
def single_unlabeled_to_labeled(input_path, output_path):
    # 加载数据
    total_data = load_pickle(input_path)
    # 创建标签
    labels = [[data[0], 1] for data in total_data]
    # 按qid和标签排序
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    # 保存标注后的数据
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    # 处理staqc Python数据
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 处理staqc SQL数据
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 处理大语料 Python数据
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 处理大语料 SQL数据
    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    # 将大语料 SQL 和 Python 单数据的未标注数据转换和保存为带标签数据
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)