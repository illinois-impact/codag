#!/usr/bin/python
import sys
import zlib
#import cv2
import os
import multiprocessing
import mmap
import zstandard as zstd
import mmap
import numpy as np
import binascii

MIN_MATCH = 1;
#N_PROCS = 80-4;
N_PROCS = 3;
WARP_SIZE = 32;
#PATH = "/home/zaid/.local/share/Steam/steamapps/common/dds/"
#PATH = "./data/"
#PATH = "/nvme0/deflate_data/input3/"
PATH = "/home/jpark346/compressor/input/"

#PATH = "/mnt/707f56b3-4a3c-4769-aaea-6f3b497651dc/zaid/datasets/fin_textures/"
#PATH = "./imgs/"
#PATH = "/home/zaid/nfs/datasets/new_textures/"

def zlib_compress(data):
    zc = zlib.compress(data, level=9);
    return ["zlib size: " + str(len(zc))];

level = 22;
def zstd_compress(data):
    cctx = zstd.ZstdCompressor(level=level);
    zstdc = cctx.compress(data);
    return ["zstd size: " + str(len(zstdc))];


#strides = [4, 8, 16, 32];
#chunks = [4096, 32*1024, 64*1024];
strides = [32];
#chunks = [128*1024];

chunks = [1024 * 128];

def zlib_chunk_compress(data, str_id):

    np_arr = np.frombuffer(data, dtype=np.uint8);
    data_copy = np.copy(np_arr);
    ret = [];
    f_name = "/home/jpark346/compressor/output/" + str_id + "_comp.bin"
    output_f_test2 = open(f_name, 'wb');
    col_len_f = open("/home/jpark346/compressor/output/" + str_id +"_col_len.bin", 'wb');
    blk_off_f = open("/home/jpark346/compressor/output/" + str_id+"_blk_offset.bin", 'wb');
 

    k = chunks[0]

    l_data = len(data_copy);

    l_data_b = l_data.to_bytes(8, byteorder='little');
    blk_off_f.write(l_data_b);

    pad_needed = int(((int((l_data+k-1)/k)) * k) - l_data);

    nz = np.zeros(pad_needed, dtype=np.uint8);
    data_copy_n = np.concatenate((data_copy, nz));
    n_chunks = int(len(data_copy_n)/k);

    print("n chunks: ", n_chunks)
    #64bit chunk siz,e #64 num of chunks,  col_len (64bits), input stream 

    chunk_size = chunks[0]
    chunk_size_b = chunk_size.to_bytes(8, byteorder='little')
   # output_f_test2.write(chunk_size_b)
    n_chunks_b = n_chunks.to_bytes(8, byteorder='little')
   # output_f_test2.write(n_chunks_b)

    chunk_size_array = []
    chunk_data_array = []

    i = strides[0]
    for p in range(n_chunks):
        start = p * k;
        end = (p+1) * k;
        cur_data = data_copy_n[start:end];
        data_copy_r = cur_data.reshape((-1,i));

        zc = zlib.compress(data_copy_r, level=9);
        if p == 0:
            print('cur chunk: ', binascii.hexlify(zc))
        pad_n = int((len(zc)+7)/8) * 8
        #print("pad n: ", pad_n)
        cur_chunk_size = pad_n
        cur_chunk_size_b = cur_chunk_size.to_bytes(8, byteorder='little')
        
        #if(p % 1000 == 0):
        print("c: ", p)

        #output_f_test2.write(cur_chunk_size_b)


        chunk_size_array.append(cur_chunk_size)
        chunk_data_array.append(zc)

    for chunk in range(n_chunks):
        cur_chunk = chunk_data_array[chunk]
        output_f_test2.write(cur_chunk)
        pad_n =  int((len(cur_chunk)+7)/8)*8 - len(cur_chunk)
        #print("len cur chunk: " ,len(cur_chunk), "pad n: ", pad_n)
        for i in range(pad_n):
            z = 0;
            zb = z.to_bytes(1, byteorder='little')
            output_f_test2.write(zb)

        #if chunk == 0:
         #   print('cur chunk: ', binascii.hexlify(cur_chunk))


    b_off = 0
    b_off_b = b_off.to_bytes(8, byteorder='little')
    blk_off_f.write(b_off_b)
    for chunk in range(n_chunks):
        b_off = b_off + chunk_size_array[chunk]
        b_off_b = b_off.to_bytes(8, byteorder='little')
        blk_off_f.write(b_off_b)
        col_len_f.write(chunk_size_array[chunk].to_bytes(8, byteorder='little'))
        #print("col len: ", chunk_size_array[chunk])

  


def run(id, files):
  #  output_f = open("output/"+str(id), 'w');
    #output_err = open("output/"+str(id)+"_err", 'w');
    #sys.stderr = output_err;
    separator = '\t';
    files.sort(key=lambda f: os.stat(f).st_size, reverse=True);
    for file in files:
        filenames = os.path.basename(file)
        filename = filenames.split(".")[0]
        print("file: ", filename)
        if (file[-4:] == ".dat"):
            continue
        line = [file];

        f = open(file, "r+b");
        dmap = mmap.mmap(f.fileno(), 0);
        orig_size = os.path.getsize(file);
        line.append("orig_size: " + str(orig_size));

        b = bytes(dmap);
        sys.stderr.write(file + " START ******************************************************\n");

        zlib_chunk_compress(b, filename)
        f.close();
        sys.stderr.write(file + " END ******************************************************\n");


        #line.append(jpg_size);
        #pixel_size = img.shape[0]*img.shape[1]*img.shape[2];
        #line.append(pixel_size);
        #line = line + method1(img);
        #print("here1")
        #line = line + method2(img);
        #print("here2")
        #line = line + method3(img);
        #print("here3")
        #line = line + method4(img);
        #print("here4")
        #line = line + method5(img);
        #print("here5")

        #break;
    #output_err.close();

if __name__ == '__main__':
    files = [(PATH+f) for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))];
    print("files: ", files)
    n_files = len(files);
    print("num files", n_files);
    print(files);
    n_iters = n_files/N_PROCS;
    n_rem = n_files%N_PROCS;

    parent_conns = [None] * N_PROCS;
    child_conns = [None] * N_PROCS;
    processes = [None] * N_PROCS;
   
    #files = [sys.argv[1]]
    #print("after file: ", files)
    #print("file: ", sys.argv[1])
    #processes[0] = multiprocessing.Process(target=run, args=(0, files)) 
    #processes[0].start()
    #processes[0].join()

    

    for i in range(N_PROCS):
        begin = int(i*n_iters);
        end = int((i+1)*n_iters);
        processes[i] = multiprocessing.Process(target=run, args=(i, files[begin:end]));
        processes[i].start();
    for i in range(N_PROCS):
        processes[i].join();

   # comp_file = open("output/comp", 'r+b');
    #compdmap = mmap.mmap(comp_file.fileno(), 0);
    #compb = bytes(compdmap);
    #zout = zlib.decompress(compb);

   # print("out data");


    #out_file = open("output/test",'wb');
    #out_file.write(zout);
