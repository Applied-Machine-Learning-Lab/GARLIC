# -*- coding: utf-8 -*-
# @Time : 2023/3/30 9:09
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : BDHconversion.py
# @Software: PyCharm


def dec_to_bin(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	return format(int(i),'b')#08b


def dec_to_bnr(i: int, lenth: int = 1) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	dec = int(i)
	digits = (len(bin(dec)) - 3 + 1) if dec < 0 else (len(bin(dec)) - 2)
	if digits >= lenth:
		lenth = digits
	pattern = f"{dec & int('0b' + '1' * lenth, 2):0{lenth}b}"
	return "".join(code for code in pattern)

def dec_to_hex(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	if i.startswith("-"):
		i = i.replace('-', '').replace(' ', '')
		return "-" + str(hex(int(i)))[2:]
	else:
		return str(hex(int(i)))[2:]

def bin_to_dec(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	return str(int(str(i), 2))

def bin_to_bnr(i: str) -> str:
	return dec_to_bnr(bin_to_dec(i))

def bin_to_hex(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	if i.startswith("-"):
		i = i.replace('-', '').replace(' ', '')
		return "-" + str(hex(int(i, 2)))[2:]
	else:
		return str(hex(int(i, 2)))[2:]

def bnr_to_dec(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	for num in i:
		if num not in ["0", "1"]:
			raise ValueError("Not bin str")
	if i.startswith("0"):
		dec = int(i, 2)
	else:
		dec = int(i[1:], 2) - 0x01
		dec = -(~dec & int("0b" + "1" * (len(i) - 1), 2))
	return str(dec)

def bnr_to_bin(i: str) -> str:
	return dec_to_bin(bnr_to_dec(i))

def bnr_to_hex(i: str) -> str:
	return dec_to_hex(bnr_to_dec(i))

def hex_to_dec(i: str) -> str:
	if not isinstance(i, str):
		raise TypeError("Not str input")
	return str(int(i, 16))

def hex_to_bin(i: str) -> str:
	return dec_to_bin(hex_to_dec(i))

def hex_to_bnr(i: str) -> str:
	return dec_to_bnr(hex_to_dec(i))

import struct
def float_to_hex(i: str) -> str:
	f = float(i)
	h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
	return str(h)[2:]

if __name__ == "__main__":
    print(dec_to_bin("10"))
    print(dec_to_bin("-10"))

    print(dec_to_bnr("10"))
    print(dec_to_bnr("-10"))

    print(dec_to_hex("10"))
    print(dec_to_hex("-10"))

    print(bin_to_dec("0101"))
    print(bin_to_dec("-0101"))

    print(bin_to_bnr("1010"))
    print(bin_to_bnr("-1010"))

    print(bin_to_hex("1010"))
    print(bin_to_hex("-1010"))

    print(bnr_to_dec("010011"))
    print(bnr_to_dec("1010011"))

    print(bnr_to_hex("10100"))
    print(bnr_to_hex("01001"))

    print(hex_to_dec("a"))
    print(hex_to_dec("-a"))

    print(float_to_hex("17.5"))
    print(float_to_hex("-17.5"))