import shutil
import os

path = 'D:/MSRDailyActivity3D'
for filename in os.listdir(path):
    if filename.endswith('.txt'):
        file_path = os.path.join(path, 'skeleton')
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        shutil.copy(os.path.join(path,filename), file_path)


# def find(obj):
#     if obj.endswith(".txt"):  # endswith()  判断以什么什么结尾
#         print(obj)
#
#
# def print_list_dir(dir_path):
#     dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
#     for file in dir_files:
#         file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
#         if os.path.isfile(file_path):  # 如果是文件，就打印这个文件路径
#             find(file_path)
#         if os.path.isdir(file_path):  # 如果目录，就递归子目录
#             print_list_dir(file_path)
#
#
# if __name__ == '__main__':
#     dir_path = 'D:\Python 电子书'
#     print_list_dir(dir_path)
