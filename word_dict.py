import pickle

# 从两个语料库中提取词汇表
def get_vocab(corpus1, corpus2):
    word_vocab = set()
    for corpus in [corpus1, corpus2]:
        for i in range(len(corpus)):
            word_vocab.update(corpus[i][1][0])  # 更新词汇表，加入context中第一个部分的单词
            word_vocab.update(corpus[i][1][1])  # 更新词汇表，加入context中第二个部分的单词
            word_vocab.update(corpus[i][2][0])  # 更新词汇表，加入code部分的单词
            word_vocab.update(corpus[i][3])     # 更新词汇表，加入query部分的单词
    print(len(word_vocab))  # 打印词汇表的长度
    return word_vocab  # 返回词汇表

# 从pickle文件中加载数据
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# 处理词汇表，将两个语料库中的词汇合并并过滤掉已有的词汇
def vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))  # 读取文件1内容并转换为集合
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())  # 读取文件2内容并转换为列表

    word_set = get_vocab(total_data2, total_data2)  # 从语料库中提取词汇表

    excluded_words = total_data1.intersection(word_set)  # 找出词汇表中已有的词汇
    word_set = word_set - excluded_words  # 从词汇表中去除已有的词汇

    print(len(total_data1))  # 打印已有词汇表的长度
    print(len(word_set))  # 打印新词汇表的长度

    with open(save_path, 'w') as f:
        f.write(str(word_set))  # 将新词汇表保存到文件中

if __name__ == "__main__":
    # Python相关文件路径
    python_hnn = './data/python_hnn_data_teacher.txt'  # Python HNN数据文件路径
    python_staqc = './data/staqc/python_staqc_data.txt'  # Python STAQC数据文件路径
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'  # Python词汇字典文件路径

    # SQL相关文件路径
    sql_hnn = './data/sql_hnn_data_teacher.txt'  # SQL HNN数据文件路径
    sql_staqc = './data/staqc/sql_staqc_data.txt'  # SQL STAQC数据文件路径
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'  # SQL词汇字典文件路径

    # 待处理的语料路径
    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'  # 未标注的SQL STAQC数据文件路径
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'  # 未标注的SQL大语料数据文件路径
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'  # 要保存的新SQL词汇字典文件路径

    # 处理词汇表，生成新的SQL词汇字典
    vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)