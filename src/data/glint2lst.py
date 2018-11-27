
import sys
import os
import numpy as np
import argparse
import pickle
import numpy.random as npr
import string
import matplotlib.pyplot as plt


def get_fromlmk():
    input_dir = sys.argv[1]
    targets = sys.argv[2]
    targets = targets.strip().split(',')
    lmap = {}

    for ds in targets:
        #image_dir = os.path.join(input_dir, ds)
        lmk_file = os.path.join(input_dir, "%s_lmk"%(ds))
        if not os.path.exists(lmk_file):
            lmk_file = os.path.join(input_dir, "%s_lmk.txt"%(ds))
            if not os.path.exists(lmk_file):
              continue
        #print(ds)
        idx = 0
        for line in open(lmk_file, 'r'):
            idx+=1
            vec = line.strip().split(' ')
            assert len(vec)==12 or len(vec)==11
            image_file = os.path.join(input_dir, vec[0])
            assert image_file.endswith('.jpg')
            vlabel = -1 #test mode
            if len(vec)==12:
                label = int(vec[1])
                if label in lmap:
                    vlabel = lmap[label]
                else:
                    vlabel = len(lmap)
                    lmap[label] = vlabel
                lmk = np.array([float(x) for x in vec[2:]], dtype=np.float32)
            else:
                lmk = np.array([float(x) for x in vec[1:]], dtype=np.float32)
            lmk = lmk.reshape( (5,2) ).T
            lmk_str = "\t".join( [str(x) for x in lmk.flatten()] )
            print("0\t%s\t%d\t0\t0\t0\t0\t%s"%(image_file, vlabel, lmk_str))
            #if idx>10:
            #  break

def get_args():
    parser = argparse.ArgumentParser(description="generate img list")
    parser.add_argument('--img-dir',type=str,dest='img_dir',default='./img_dir',\
                        help='the input directory saved images')
    parser.add_argument('--lmk-list',type=str,dest='lmk_list',default='msra,asian',\
                        help='the input list saved images')
    parser.add_argument('--out-file',type=str,dest='out_file',default='./output.lst',\
                        help='the output txt saved images')
    parser.add_argument('--base-id',type=int,dest='base_id',default=None,\
                        help='the image label first one')
    parser.add_argument('--save-label',type=int,dest='save_label',default=0,\
                        help='whether the img is prison')
    parser.add_argument('--cmd-type',type=str,dest='cmd_type',default=None,\
                        help='which to run, gen_from_dir; gen_from_lmk; hist')
    parser.add_argument('--prison-data',type=int,dest='prison_data',default=0,\
                        help=' if prison data ')
    parser.add_argument('--file1-in',type=str,dest='file1_in',default=None,\
                        help='inout file1 ')
    parser.add_argument('--file2-in',type=str,dest='file2_in',default=None,\
                        help='inout file2')
    parser.add_argument('--hist-max',type=float,dest='hist_max',default=None,\
                        help='histgram max range num')
    parser.add_argument('--hist-bins',type=int,dest='hist_bins',default=20,\
                        help='histgram annotate x num bins')
    return parser.parse_args()


def merge2trainfile(file1,file2,file_out):
    '''
    file1: saved image  paths
    file2: saved image paths
    return: "file1 file2" 
    '''
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f_out = open(file_out,'w')
    id_files = f1.readlines()
    imgs = f2.readlines()
    for line_one in id_files:
        f_out.write(line_one.strip())
        f_out.write("\n")
    for line_one in imgs:
        f_out.write(line_one.strip())
        f_out.write("\n")
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def generate_list_from_dir(dirpath,out_file,base_id,prison_data,face_align,save_label):
    '''
    dirpath: saved images path
            "dirpath/id_num/image1.jpg"
    return: images paths txtfile
            "id_num/img1.jpg"
    base_id: the fist label
            "real_id + base_id"
    '''
    f_w = open(out_file,'w')
    files = os.listdir(dirpath)
    total_ = len(files)
    idx =0
    file_name = []
    total_cnt = 0
    id_files = np.sort(files)
    label_dict = dict()
    Save_Label = save_label
    for label,file_cnt in enumerate(id_files):
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        if base_id is None:
            label = int(file_cnt.strip())
        else:
            label = label+base_id
        if Save_Label:
            id_num = file_cnt[2:]
            label_dict[id_num] = label
        if prison_data==1 and len(imgs)>200 :
            keep_idx = npr.choice(len(imgs), size=200, replace=False)
            imgs = [imgs[i] for i in keep_idx]
        for img_one in imgs:
            if prison_data==1 and len(img_one)<=8:
                continue
            img_path = os.path.join(img_dir,img_one)
            total_cnt+=1
            f_w.write("{} {} {}\n".format(face_align,img_path,label))
    print("total id ",len(files))
    print("total img ",total_cnt)
    if Save_Label:
        f_label = open(out_file[:-4]+".pkl",'wb')
        pickle.dump(label_dict,f_label)
        f_label.close()
    f_w.close()

def plt_histgram(file_in,file_out,distance,num_bins=20):
    '''
    file_in: saved train img path  txt file: /img_path/0.jpg  1188
    file_out: output bins and 
    '''
    out_name = file_out
    input_file=open(file_in,'r')
    out_file=open(file_out,'w')
    data_arr=[]
    print(out_name)
    out_list = out_name.strip()
    out_list = out_list.split('/')
    out_name = "./output/"+out_list[-1][:-4]+".png"
    print(out_name)
    id_dict_cnt = dict()
    for line in input_file.readlines():
        line = line.strip()
        line_splits = line.split(' ')
        key_name=string.atoi(line_splits[-1])
        cur_cnt = id_dict_cnt.setdefault(key_name,0)
        id_dict_cnt[key_name] = cur_cnt +1
    for key_num in id_dict_cnt.keys():
        data_arr.append(id_dict_cnt[key_num])
    data_in=np.asarray(data_arr)
    if distance is None:
        max_bin = np.max(data_in)
    else:
        max_bin = distance
    datas,bins,c=plt.hist(data_in,num_bins,range=(0.0,max_bin),normed=0,color='blue',cumulative=0)
    #a,b,c=plt.hist(data_in,num_bins,normed=1,color='blue',cumulative=1)
    plt.title('histogram')
    plt.savefig(out_name, format='png')
    plt.show()
    for i in range(num_bins):
        out_file.write(str(datas[i])+'\t'+str(bins[i])+'\n')
    input_file.close()
    out_file.close()

if __name__ == '__main__':
    args = get_args()
    dirpath = args.img_dir
    out_file = args.out_file
    base_id = args.base_id
    prison_data = args.prison_data
    cmd_type = args.cmd_type
    file1_in = args.file1_in
    file2_in = args.file2_in
    face_align = 1
    hist_bins = args.hist_bins
    hist_max = args.hist_max
    file_out = args.out_file
    save_label = args.save_label
    if cmd_type == 'gen_from_dir':
        generate_list_from_dir(dirpath,out_file,base_id,prison_data,face_align,save_label)
    elif cmd_type == 'merge':
        merge2trainfile(file1_in,file2_in,out_file)
    elif cmd_type == 'hist':
        plt_histgram(file1_in,file_out,hist_max,hist_bins)