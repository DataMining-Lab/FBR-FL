import openpyxl

# 示例数组
data_array = [1, 2, 3, 4, 5]

# 创建一个新的 Excel 文件
workbook = openpyxl.Workbook()
sheet = workbook.active

# 将数组的每个元素写入每一列
for index, value in enumerate(data_array, start=1):
    sheet.cell(row=index, column=1, value=value)

# 保存文件
workbook.save('array_to_excel.xlsx')