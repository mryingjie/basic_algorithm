arr = [42,45,6,6,6,6,42]
# arr = [32,4,321,2,13,7]


#划分 给定一个数组索引区域，和此区域中的某个数的索引， 将大于这个值的数放在这个值的右边 小于的相反
def partition(arr,head,tail):
    pivot  = arr[head]
    left = head
    right = tail
    #此处使用指针交换的方法来实现分治算法
    while left != right:
        #当元素的值和基准元素的值相等时不移动这个元素 直到头尾相等时结束并记录这个索引值
        while arr[right] >= pivot and right > left:
            right = right - 1
        while arr[left] <= pivot and left < right:
            left = left + 1
        #交换头尾的位置
        if left< right:
            arr[left],arr[right] = arr[right],arr[left]
    # 外循环要结束时 交换基准元素和 最后的那个元素的位置
    arr[head], arr[left] = arr[left], arr[head]
    return left

def quick_sort_base(arr,head,tail):
    if head>= tail:
        return
    #第一次划分，使用第一各元素的位置为基准元素 并 得到划分后的基准元素的位置 这个基准元素的位置已经确定 递归时不必再参与排序
    index = partition(arr,head,tail)
    #递归划分 左右两边的元素
    quick_sort_base(arr,head,index - 1)
    quick_sort_base(arr,index + 1,tail)

def quick_sort(arr):
    head = 0
    tail = len(arr) - 1
    quick_sort_base(arr,head,tail)

quick_sort(arr)
print(arr)