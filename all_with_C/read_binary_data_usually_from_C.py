from IPython import embed
import os
import sys
import array
import numpy as np


def read_binary_data(name, size_of_one_number, endian="Little"):                                                  
    system_endian = sys.byteorder.capitalize()
    count = int(os.stat(name).st_size / size_of_one_number)
    print("Will reading %s from your file according the size you given (%s)"%(count, size_of_one_number))
    with open(name, 'rb') as f:
        result = array.array('f')     #TODO
        result.fromfile(f, count)
        #embed()
        if endian != system_endian: result.byteswap()
    return np.array(result).astype(np.float)


if __name__ == "__main__":
    print("Start.")
    filename = "green_binary.out"
    size_of_one_number = 4     #TODO
    data = read_binary_data(filename, size_of_one_number)

                                                                                                                  

