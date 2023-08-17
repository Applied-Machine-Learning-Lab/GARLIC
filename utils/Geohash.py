# -*- coding: utf-8 -*-
# @Time : 2023/3/29 22:39
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : Geohash.py
# @Software: PyCharm

class Geohash:
    def __init__(self):
        self._all_ = ['encode', 'decode', 'bbox', 'neighbors']
        _base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        # 10进制和32进制转换，32进制去掉了ailo
        self._decode_map = {}  # 解码表
        self._encode_map = {}  # 编码表
        for i in range(len(_base32)):
            self._decode_map[_base32[i]] = i  # _decode_map[字符]=数字
            self._encode_map[i] = _base32[i]  # _encode_map[数字]=字符

    # 编码函数
    # 将经纬度 编码 为字符串
    def encode(self, lat, lon, precision=12):
        """
        :param lat: 纬度
        :param lon: 经度
        :param precision: 精度
        :return:
        """
        lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
        geohash = []  # 地理位置哈希表
        code = []
        j = 0
        c_out = []
        while len(geohash) < precision:
            # print(code, lat_range, lon_range, geohash)
            j += 1
            lat_mid = sum(lat_range) / 2
            lon_mid = sum(lon_range) / 2
            # 经度 -> 经度和纬度交叉放置
            if lon <= lon_mid:  # 交线位置给左下
                code.append(0)
                lon_range[1] = lon_mid
            else:
                code.append(1)
                lon_range[0] = lon_mid
            # 纬度
            if lat <= lat_mid:  # 交线位置给左下
                code.append(0)
                lat_range[1] = lat_mid
            else:
                code.append(1)
                lat_range[0] = lat_mid
            # encode  编码
            if len(code) >= 5:
                # 每5位2进制数 用1个字符表示
                geohash.append(self._encode_map[int(''.join(map(str, code[:5])), 2)])
                c_out.append(code[0])
                c_out.append(code[1])
                c_out.append(code[2])
                c_out.append(code[3])
                c_out.append(code[4])
                code = code[5:]
        return ''.join(geohash), c_out

    # 解码函数
    # 将字符串 解码 为经纬度
    def decode(self, geohash):
        lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
        is_lon = True
        for letter in geohash:  # 取每个字符
            # 每个字符都对应着5位二进制数, 前面不够的用0补齐
            s = str(bin(self._decode_map[letter]))[2:]
            code = s.rjust(5, '0')  # 不够5位，前面用0补齐
            for bi in code:
                if is_lon and bi == '0':
                    lon_range[1] = sum(lon_range) / 2
                elif is_lon and bi == '1':
                    lon_range[0] = sum(lon_range) / 2
                elif (not is_lon) and bi == '0':
                    lat_range[1] = sum(lat_range) / 2
                elif (not is_lon) and bi == '1':
                    lat_range[0] = sum(lat_range) / 2
                is_lon = not is_lon  # 经度和纬度是交替出现的
        return sum(lat_range) / 2, sum(lon_range) / 2

    def decode_c(self, c_code):
        geohash = []
        while len(c_code) >= 5:
            # 每5位2进制数 用1个字符表示
            geohash.append(self._encode_map[int(''.join(map(str, c_code[:5])), 2)])
            c_code = c_code[5:]
        return self.decode(geohash)

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}  # 字典
        self.end = -1  # 结束标志

    def insert(self, word):
        """  将单词插入到字典树中
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root  # 从根结点开始遍历
        for c in word:  # 把每个字符都放入到 字典树 中
            if c not in curNode:  # 字符c不在当前结点中，则创建新分支。
                curNode[c] = {}
            curNode = curNode[c]  # 下一个结点
        curNode[self.end] = True  # 结束结点用True标记

    def startsWith(self, prefix):
        """  按照前缀开始搜索
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curNode = self.root
        for c in prefix:
            if c not in curNode:
                return []
            curNode = curNode[c]
        # print(curNode)
        path = prefix
        paths = []  # 找到所有的相同前缀的路径，返回
        self.find_path_all(curNode, path, paths)
        return paths

    # 从树中找出从起始顶点到终止顶点的所有路径
    def find_path_all(self, root, path, paths):
        """ 利用深度优先搜索
        :param root: 当前顶点
        :param path: 当前路径
        :param paths: 所有路径
        :return:
        """
        if self.end in root:  # 判断是否搜索到末尾了。
            paths.append(path)
            return
        for v in root:
            # 构造下次递归的父路径
            path += v  # 加入顶点
            self.find_path_all(root[v], path, paths)  # 递归
            path = path[:-1]  # 回溯


if __name__ == "__main__":
    geohash = Geohash()
    str, ori = geohash.encode(lon=120.13345, lat=30.12342, precision=12)
    print(str)
    print(ori)
    lat, lon = geohash.decode(str)
    print(lon, lat)
    lat, lon = geohash.decode_c(ori)
    print(lon, lat)