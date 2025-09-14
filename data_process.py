import pandas as pd


def move_columns_to_end(file_path, columns_to_move, output_path=None):
    """
    Moves specified columns to the end of a DataFrame and saves the result to a new CSV file.

    Args:
        file_path (str): The path to the source CSV file.
        columns_to_move (list): A list of column names to move to the end.
        output_path (str, optional): The path to save the new CSV file.
                                     If None, it overwrites the original file.
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查所有待移动的列是否存在
        missing_columns = [col for col in columns_to_move if col not in df.columns]
        if missing_columns:
            print(f"警告: 找不到以下列，它们将被忽略: {missing_columns}")
            # 过滤掉不存在的列
            columns_to_move = [col for col in columns_to_move if col in df.columns]

        # 分离出要移动的列和其余的列
        cols_to_keep = [col for col in df.columns if col not in columns_to_move]

        # 重新排列DataFrame，将要移动的列放在最后
        df_reordered = df[cols_to_keep + columns_to_move]

        # 确定保存路径
        if output_path is None:
            output_path = file_path

        # 保存新的CSV文件
        df_reordered.to_csv(output_path, index=False)
        print(f"文件处理完成！已将指定的列移动到末尾并保存到 '{output_path}'。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。请检查文件路径是否正确。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


# --- 使用示例 ---
# 待处理的CSV文件路径
csv_file = 'train1.csv'

# 你想移动到末尾的列名列表
columns_to_move = [
    'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR',
    'P1_PP02D', 'P1_PP02R', 'P1_STSP', 'P2_ATSW_Lamp',
    'P2_AutoGO', 'P2_Emerg', 'P2_MASW', 'P2_MASW_Lamp',
    'P2_ManualGO', 'P2_OnOff', 'P2_TripEx', 'Attack'
]

# 调用函数
move_columns_to_end(file_path=csv_file, columns_to_move=columns_to_move)

# 如果你想保存到新的文件，而不是覆盖原文件，可以指定 output_path 参数
# 例如: move_columns_to_end(csv_file, columns_to_move, output_path='new_file.csv')

# P1_PP01AD、P1_PP01AR、P1_PP01BD、P1_PP01BR、P1_PP02D、P1_PP02R、P1_SOL01D、P1_SOL03D、P1_STSP、P2_ATSW_Lamp、P2_AutoGO
# P2_Emerg、P2_MASW、P2_MASW_Lamp、P2_ManualGO、P2_OnOff、P2_TripEx  17个执行器、69个传感器