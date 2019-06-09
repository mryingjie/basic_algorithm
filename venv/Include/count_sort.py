#计数排序适用于对数据范围可穷举的一组数据进行排序  比如10以内的整数
arr = [3,5,6,2,3,7,8,1,9,7,4,8]

def count_sort(arr):
    # 获取列表最大值
    max_value = max(arr)
    # 根据最大值创建统计数组
    count_arr = [0] * (max_value + 1)
    # 遍历数组
    for i in arr:
        count_arr[i] = count_arr[i] + 1
    # 再次遍历数组输出排序结果
    result_arr = []
    #遍历range时 左闭右开
    for i in range(len(count_arr)):
        for j in range(count_arr[i]):
            result_arr.append(i)
    return result_arr

print(count_sort(arr))
