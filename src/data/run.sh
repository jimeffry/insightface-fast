#generate Ms-cleb-1m 
#python glint2lst.py  --img-dir /home/lxy/Downloads/DataSet/Ms-1M-Celeb/train/ --out-file ./output/ms_celeb_1m.txt --prison-data 0 --cmd-type gen_from_dir
#generate prison data
#python glint2lst.py  --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_v1/ --out-file ./output/prison_train.txt --prison-data 1 --base-id 85164 --save-label 1 --cmd-type gen_from_dir
#merge 2 files
#python glint2lst.py  --file1-in ./output/ms_celeb_1m.txt --file2-in ./output/prison_train.lst --out-file ./output/ms_prison_v2.lst --cmd-type merge

##generate .rec and .idx
#python face2rec2.py  --prefix /home/lxy/Downloads/DataSet/Face_reg/prison_ms/
#python face2rec2.py  --prefix /home/lxy/Downloads/DataSet/Ms-Celeb-1m/prison_train/
## plot histgram dataset distribute
#python glint2lst.py --file1-in ./output/ms_prison_v2.lst --out-file ./output/dataset_dis2.txt --hist-bins 50 --cmd-type hist
## merge 2 datasets
#python dataset_merge.py --include /home/lxy/Downloads/DataSet/Ms-Celeb-1m/prison_train/,/home/lxy/Downloads/DataSet/Ms-Celeb-1m/faces_ms1m_112x112 \
#        --output /home/lxy/Downloads/DataSet/Ms-Celeb-1m/prison_ms/ --param1 0.0

#python unpack_mxrecord.py 