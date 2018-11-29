# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/11/29 11:09
#project: Face recognize
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description:
####################################################
import mxnet as mx 

def plot_model(sym,data_shape,img_path):
    '''
    sym: model symbole
    data_shape: input data shape
    img_path: save path
    '''
    data_dict = {'data':data_shape}
    net_show = mx.viz.plot_network(symbol=sym,shape=data_dict,title='model',save_format='png')
    net_show.render(filename='insightface_train',cleanup=True)