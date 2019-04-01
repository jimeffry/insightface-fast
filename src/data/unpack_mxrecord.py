import mxnet as mx
import argparse
import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import os
import string

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--show_fg', default=False, type=bool,
                        help='if to show the saved tfrecords')
    parser.add_argument('--base_dir', default="/home/lxy/Downloads/DataSet/Ms-1M-Celeb/train", type=str,
                        help='images saved dir')
    args = parser.parse_args()
    return args


def saved_img(img,base_dir,label,cnt):
    dir_path = os.path.join(base_dir,label)
    fix = str(cnt)+".jpg"
    #img_path = os.path.join(dir_path,fix)
    if os.path.exists(dir_path):
        img_path = os.path.join(dir_path,fix)
        cv2.imwrite(img_path,img)
    else:
        os.makedirs(dir_path)
        img_path = os.path.join(dir_path,fix)
        cv2.imwrite(img_path,img)

    

def mx2tfrecords_old(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        encoded_jpg_io = io.BytesIO(img)
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img_raw = img.tobytes()
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    #img = tf.reshape(img, shape=(112, 112, 3))
    #r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    #r, g, b = img[:,:,0],img[:,:,1],img[:,:,2]
    #img = tf.concat([b, g, r], axis=-1)
    #img[:,:,0],img[:,:,1],img[:,:,2] = b,g,r
    #img = tf.cast(img, dtype=tf.float32)
    #img = tf.subtract(img, 127.5)
    #img = tf.multiply(img,  0.0078125)
    #img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


if __name__ == '__main__':
    # # define parameters
    id2range = {}
    data_shape = (3, 112, 112)
    args = parse_args()
    base_dir = args.base_dir
    if args.show_fg:
        imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        print(header.label)
        imgidx = list(range(1, int(header.label[0])))
        seq_identity = range(int(header.label[0]), int(header.label[1]))
        print("label",int(header.label[0]), int(header.label[1]))
        
        for identity in seq_identity:
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            id2range[identity] = (a, b)
            if identity %10000 ==0:
                print(identity)
        #print('id2range', len(id2range))

        # # generate tfrecords
        mx2tfrecords(imgidx, imgrec, args)
    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        # training datasets api config
        tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
        #dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = tf.contrib.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=30000)
        dataset = dataset.batch(1)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        # begin iteration
        sess.run(iterator.initializer)
        images, labels = sess.run(next_element)
        cnt_dict = dict()
        i = 0
        #for i in range(10):
        while images is not None:
            print("num ----",i)
            #sess.run(iterator.initializer)
            #while True:
            i+=1
            try:
                #images, labels = sess.run(next_element)
                print("label", labels[0, ...],images.shape)
                img = cv2.cvtColor(images[0,...], cv2.COLOR_RGB2BGR)
                label_s = str(labels[0, ...])
                cnt_dict.setdefault(label_s,0)
                cnt_dict[label_s]+=1
                saved_img(img,base_dir,label_s,cnt_dict[label_s])
                #cv2.imshow('test', img)
                #cv2.waitKey(1000)
                images, labels = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        
