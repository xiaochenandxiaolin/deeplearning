import os
from config import path,class_data
import json


def json_data(file_dir):
# file_dir=path
    # a,b,c,d,e=[],[],[],[],[]
    root_l,dirs_l,file_l,num_l=[],[],[],[]
    for root, dirs, files in os.walk(file_dir):  
        dirs_l.append(dirs)
        root_l.append(root)
        file_l.append(files)
    # print(b)
    dir_list=dirs_l[0]
    for i in range(len(dir_list)):
        src=root_l[0]+'/'+dir_list[i]
        dst=root_l[0]+'/'+str(i)
        os.rename(src,dst)
        num_l.append(i)
        os.rename(src,dst)
    class_json=dict(zip(num_l,dir_list))
    with open (class_data,'w') as f:
        f.write(json.dumps(class_json,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    json_data(path)