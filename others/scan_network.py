#!/usr/bin/python
#_*_ coding:utf8 _*_
import netifaces,nmap
from IPython import embed

def get_gateways():
    tmp = netifaces.gateways()
    return tmp['default'][netifaces.AF_INET][0]

def get_ip_lists(ip):
    ip_lists = []
    for j in range(1, 256):
        for i in range(1, 256):
            ip_lists.append('.'.join(ip.split('.')[:-2] + [str(j),str(i)]))
    return ip_lists

def main(ip=None):
    ip=get_gateways()
    ip_lists=get_ip_lists(ip)
    nmScan = nmap.PortScanner()
    temp_ip_lists = []
    hosts = ip[:-1]+'0/24'
    ret = nmScan.scan(hosts=hosts, arguments='-sP')
    print('扫描时间：'+ret['nmap']['scanstats']['timestr']+'\n命令参数:'+ret['nmap']['command_line'])
    for ip in ip_lists:
        if ip not in ret['scan']:
            temp_ip_lists.append(ip)
        else:
            print('%s已扫描到主机，主机名：'%ip+ret['scan'][ip]['hostnames'][0]['name'])
    print(str(hosts) +' 网络中的存活主机:')
    for ip in temp_ip_lists:ip_lists.remove(ip)
    for ip in ip_lists:print(ip)

if __name__ == '__main__':
    main()

