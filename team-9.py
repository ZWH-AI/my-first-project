#  team 9 members: 张文华 罗兰兰 胡诗音 张榆 徐瑞
import multiprocessing
from multiprocessing import Process
import time
import random
from matrix_multiply import matrix_multiplication_non_multiprocess
from matrix_multiply import matrix_multiplication_with_multiprocess
from matrix_multiply import matrix_compare

# 输入随机矩阵的行列数
def input_matrix_dimensions():
    while True:
        try:
            rows_A = int(input("请输入矩阵A的行数（至少为2）: "))
            cols_A = int(input("请输入矩阵A的列数（至少为2，且必须等于矩阵B的行数）: "))
            rows_B = int(input("请输入矩阵B的行数（必须等于矩阵A的列数，且至少为2）: "))
            cols_B = int(input("请输入矩阵B的列数（至少为2）: "))

            # 检查行数和列数是否至少为2
            if rows_A < 2 or cols_A < 2 or rows_B < 2 or cols_B < 2:
                print("矩阵的行数和列数都必须至少为2，请重新输入。")
                continue

                # 检查A的列数是否等于B的行数
            if cols_A == rows_B:
                return rows_A, cols_A, rows_B, cols_B
            else:
                print("矩阵A的列数必须等于矩阵B的行数，请重新输入。")
        except ValueError:
            print("输入的不是整数，请重新输入。")
        # 调用函数以获取矩阵A和B的维度

# 生成随机矩阵函数
def generate_random_matrix(rows, cols):
    def get_user_choice():
        while True:
            print("请选择要生成的矩阵类型：")
            print("1. 整数矩阵")
            print("2. 浮点数矩阵")
            choice = input("请输入选择（1 或 2）：")
            if choice in ['1', '2']:
                return int(choice)
            else:
                print("输入数据无效，请重新输入！")

    choice = get_user_choice()
    if choice == 1:  # 整数矩阵
        return [[random.randint(-10000, 10000) for _ in range(cols)] for _ in range(rows)]
    elif choice == 2:  # 浮点数矩阵
        # 注意：这里假设我们想要生成在 -10000.0 到 10000.0 之间的浮点数
        return [[random.uniform(-10000.0, 10000.0) for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("输入数据无效！")

# 切分矩阵的函数
# 第一个矩阵的列数必须等于第二个矩阵的行数
def split_matrix(matrix):
    """
    将矩阵分割为四个合理的小矩阵
    :param matrix: 待分割的矩阵
    :return: 分割后的四个小矩阵（列表的列表）
    """
    # 获取原始矩阵的行数和列数
    rows, cols = len(matrix), len(matrix[0]) #3，3

    # 向下取整，确保能够整除
    submatrix_rows = rows // 2  #1
    submatrix_cols = cols // 2  #1

    # 初始化结果列表
    submatrices = []

    # 左上矩阵
    top_left = [row[0:submatrix_cols] for row in matrix[:submatrix_rows]]
    submatrices.append(top_left)

    # 右上矩阵
    top_right = [row[submatrix_cols:] for row in matrix[:submatrix_rows]]
    submatrices.append(top_right)

    # 左下矩阵
    if (rows % 2 != 0) or (cols % 2 != 0):
        bottom_left = [row[0:submatrix_cols] for row in matrix[submatrix_rows:submatrix_rows + submatrix_rows + 1]]
        submatrices.append(bottom_left)
    else:
        bottom_left = [row[0:submatrix_cols] for row in matrix[submatrix_rows:submatrix_rows + submatrix_rows]]
        submatrices.append(bottom_left)

    # 右下矩阵
    if (rows % 2 != 0) or (cols % 2 != 0):
        bottom_right = [row[submatrix_cols:] for row in matrix[submatrix_rows:submatrix_rows + submatrix_rows + 1]]
        submatrices.append(bottom_right)
    else:
        bottom_right = [row[submatrix_cols:] for row in matrix[submatrix_rows:submatrix_rows + submatrix_rows]]
        submatrices.append(bottom_right)

    return submatrices

# 矩阵乘法函数
def matrix_multiply(A, B, result_queue,add_result_queue):
    # 检查矩阵A的列数是否与矩阵B的行数相等
    if len(A[0]) == len(B):
        # 获取矩阵A的行数和矩阵B的列数
        rows_A = len(A)
        cols_B = len(B[0])

        # 初始化结果矩阵C，其行数为A的行数，列数为B的列数
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

        # 执行矩阵乘法
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(len(B)):  # 这里的len(B)实际上是矩阵B的行数，与矩阵A的列数相同
                    C[i][j] += A[i][k] * B[k][j]

        result_queue.put(C)
        if (result_queue.qsize() == 2):
            p = Process(target=matrix_addition,
                        args=(result_queue.get(), result_queue.get(), add_result_queue))
            p.start()

# 矩阵加法函数
def matrix_addition(A, B, add_result_queue):

    # 检查两个矩阵是否为同型矩阵（即具有相同的行数和列数）
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("两个矩阵必须是同型的才能进行加法运算")

        # 初始化结果矩阵C，与A和B具有相同的形状
    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]

    # 执行矩阵加法
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
    add_result_queue.put(C)

# 多进程矩阵乘法运算函数
def Multiprocess(submatrices1, submatrices2, result_queue1,result_queue2,result_queue3,result_queue4, add_result_queue1,add_result_queue2,add_result_queue3,add_result_queue4):

    # 子矩阵乘法的多线程实现（8个进程）
    t1 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[0], submatrices2[0], result_queue1, add_result_queue1))
    t2 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[1], submatrices2[2], result_queue1, add_result_queue1))
    t3 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[0], submatrices2[1], result_queue2, add_result_queue2))
    t4 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[1], submatrices2[3], result_queue2, add_result_queue2))
    t5 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[2], submatrices2[0], result_queue3, add_result_queue3))
    t6 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[3], submatrices2[2], result_queue3, add_result_queue3))
    t7 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[2], submatrices2[1], result_queue4, add_result_queue4))
    t8 = multiprocessing.Process(target=matrix_multiply, args=(submatrices1[3], submatrices2[3], result_queue4, add_result_queue4))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()

    # 左右拼接 add1 和 add2，以及 add3 和 add4
    concat_left_right1 = [a + b for a, b in zip(add_result_queue1.get(), add_result_queue2.get())]
    concat_left_right2 = [a + b for a, b in zip(add_result_queue3.get(), add_result_queue4.get())]

    # 上下拼接上面两个结果
    concat_top_bottom = concat_left_right1 + concat_left_right2

    return concat_top_bottom

# 单进程矩阵乘法运算函数
def solo_matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("矩阵乘法维度不匹配")
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

# 多进程启动
def multiprocessRun(A, B):

    # 创建乘法进程队列
    result_queue1 = multiprocessing.Queue()
    result_queue2 = multiprocessing.Queue()
    result_queue3 = multiprocessing.Queue()
    result_queue4 = multiprocessing.Queue()

    # 创建加法进程队列
    add_result_queue1 = multiprocessing.Queue()
    add_result_queue2 = multiprocessing.Queue()
    add_result_queue3 = multiprocessing.Queue()
    add_result_queue4 = multiprocessing.Queue()

    # 切分A、B两个矩阵分别为4个小矩阵
    submatrices1 = split_matrix(A)
    submatrices2 = split_matrix(B)

    result = Multiprocess(submatrices1, submatrices2, result_queue1, result_queue2, result_queue3, result_queue4, add_result_queue1, add_result_queue2, add_result_queue3, add_result_queue4)

    return result

# 主程序
if __name__ == "__main__":

    rows_A, cols_A, rows_B, cols_B = input_matrix_dimensions()

    # 生成矩阵A和B
    A = generate_random_matrix(rows_A, cols_A)
    B = generate_random_matrix(rows_B, cols_B)

    begin_time = time.time()
    team_result = solo_matrix_multiply(A, B)
    print("小组单进程运行时间:",time.time() - begin_time)

    begin_time = time.time()
    demo_result = matrix_multiplication_non_multiprocess(A, B)
    print("对比组单进程运行时间:",time.time() - begin_time)

    print("单进程结果一致性: ",matrix_compare(team_result, demo_result))

    begin_time = time.time()
    team_result = multiprocessRun(A, B)
    print("小组多进程运行时间:",time.time() - begin_time)

    begin_time = time.time()
    demo_result = matrix_multiplication_with_multiprocess(A, B)
    print("对比组多进程运行时间:",time.time() - begin_time)

    print("多进程结果一一致性: ",matrix_compare(team_result,demo_result))





