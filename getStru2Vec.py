import pickle
import multiprocessing
from python_structured import *  # 导入处理Python结构化数据的相关函数
from sqlang_structured import *  # 导入处理SQL结构化数据的相关函数

# 多处理函数，用于并行解析Python查询数据
def multipro_python_query(data_list):
    return [python_query_parse(line) for line in data_list]

# 多处理函数，用于并行解析Python代码数据
def multipro_python_code(data_list):
    return [python_code_parse(line) for line in data_list]

# 多处理函数，用于并行解析Python上下文数据
def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result

# 多处理函数，用于并行解析SQL查询数据
def multipro_sqlang_query(data_list):
    return [sqlang_query_parse(line) for line in data_list]

# 多处理函数，用于并行解析SQL代码数据
def multipro_sqlang_code(data_list):
    return [sqlang_code_parse(line) for line in data_list]

# 多处理函数，用于并行解析SQL上下文数据
def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

# 通用解析函数，用于调用不同的解析函数对数据进行并行处理
def parse(data_list, split_num, context_func, query_func, code_func):
    pool = multiprocessing.Pool()  # 创建一个进程池
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]  # 将数据分割为多个小块
    results = pool.map(context_func, split_list)  # 并行处理上下文数据
    context_data = [item for sublist in results for item in sublist]  # 展开处理结果
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)  # 并行处理查询数据
    query_data = [item for sublist in results for item in sublist]  # 展开处理结果
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)  # 并行处理代码数据
    code_data = [item for sublist in results for item in sublist]  # 展开处理结果
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()  # 关闭进程池并等待所有进程完成

    return context_data, query_data, code_data  # 返回处理后的数据

# 主函数，用于管理整个解析流程
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)  # 从文件中加载数据

    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)  # 解析数据
    qids = [item[0] for item in corpus_lis]  # 提取qid

    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]  # 组合成最终数据

    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)  # 将结果保存到文件中

# 程序入口
if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'  # Python Staqc数据路径
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'  # Python Staqc保存路径

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'  # SQL Staqc数据路径
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'  # SQL Staqc保存路径

    # 处理Python和SQL的Staqc数据
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    # 处理Python和SQL的大语料数据
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'  # Python大语料数据路径
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'  # Python大语料数据保存路径

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'  # SQL大语料数据路径
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'  # SQL大语料数据保存路径

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)