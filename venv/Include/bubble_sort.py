

arr = [32,4,21,45,6,75,4,42,523,6,42]

#冒泡排序
def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
        #将小的数往前放
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]


bubbleSort(arr)

print(arr)