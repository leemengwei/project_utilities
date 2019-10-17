def read_depth_data(depthname, endian="little"):              
    count = int(stat(depthname).st_size / 2)
    with open(depthname, 'rb') as f:
        result = array('h')
        result.fromfile(f, count)
        if endian != system_endian: result.byteswap()
    return np.array(result).reshape(960,1280).astype(np.float)


