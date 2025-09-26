#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File       ：extract_data.py
@Author     ：zhanyoyo
@Date       ：2025/5/25 上午3:08 
@Description：\n
'''

def lab2_1():
    import re
    import csv

    # 假设 input_files 是你的输入文件列表
    input_files = [
        '/home/linke/yoyo/edgerag/edgerag_new/run_test/results_fiqa/2_1_gen.txt',
        '/home/linke/yoyo/edgerag/edgerag_new/run_test/results_fiqa/2_1_gen+load.txt',
        '/home/linke/yoyo/edgerag/edgerag_new/run_test/results_fiqa/2_1_edgerag.txt',
        '/home/linke/yoyo/edgerag/edgerag_new/run_test/results_fiqa/2_1_ours.txt'
    ]
    output_csv = "/home/linke/yoyo/edgerag/edgerag_new/output/lab2_1/result.csv"

    # 正则模式匹配 Retrieve Time: 后面的浮点数
    pattern = re.compile(r"Retrieve Time:\s*([0-9.+\-eE]+)")

    # 提取每个文件的 Retrieve Time 列表
    all_retrieve_times = []

    for file_path in input_files:
        retrieve_times = []
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    retrieve_times.append(float(match.group(1)))

        # 按照升序排列每个文件的 Retrieve Time
        retrieve_times.sort()
        all_retrieve_times.append(retrieve_times)

    # 对齐各列（填充空值，使所有列等长）
    max_len = max(len(col) for col in all_retrieve_times)
    for col in all_retrieve_times:
        while len(col) < max_len:
            col.append('')  # 空字符串填充空位

    # 转置数据，每行一个时间点，每列一个文件
    rows = list(zip(*all_retrieve_times))

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写表头：File1, File2, ...
        header = [f'File{i + 1}' for i in range(len(input_files))]
        writer.writerow(header)
        # 写数据行
        writer.writerows(rows)

    print(f"已从 {len(input_files)} 个文件中提取 Retrieve Time，并写入 {output_csv}")


def lab2_2():
    # 输入文件路径
    import re
    input_file = '/home/linke/yoyo/edgerag/edgerag_new/run_test/results_quora/2_1_ours_s3.txt'

    # 正则表达式匹配所需的三项数据
    pattern = re.compile(
        r"Retrieve Time:\s*([0-9.+\-eE]+).*Prefill Latency:\s*([0-9.+\-eE]+).*Generation Latency:\s*([0-9.+\-eE]+)")

    # 用于存储每个指标的值
    retrieve_times = []
    prefill_latencies = []
    generation_latencies = []

    # 逐行读取文件并提取数据
    with open(input_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                retrieve_times.append(float(match.group(1)))
                prefill_latencies.append(float(match.group(2)))
                generation_latencies.append(float(match.group(3)))

    # 计算平均值
    avg_retrieve_time = sum(retrieve_times) / len(retrieve_times) if retrieve_times else 0
    avg_prefill_latency = sum(prefill_latencies) / len(prefill_latencies) if prefill_latencies else 0
    avg_generation_latency = sum(generation_latencies) / len(generation_latencies) if generation_latencies else 0
    print(avg_retrieve_time)
    print(avg_prefill_latency)
    print(avg_generation_latency)

# lab2_1()
lab2_2()