# -*- coding: utf-8 -*-
import argparse

import ctypes
from ctypes import c_float

import os
from flask import Flask
from flask import jsonify
from flask import request

import base64


parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='ip help')
parser.add_argument('--port', help='port help')
args = parser.parse_args()


class XYZ(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double),
                ('z', ctypes.c_double)]


class ToPythonRect(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int),
                ('y', ctypes.c_int),
                ('w', ctypes.c_int),
                ('h', ctypes.c_int)]


class RetBox(ctypes.Structure):
    _fields_ = [('text', ctypes.POINTER(ctypes.c_int)),
                ('charNum', ctypes.c_int),
                ('cx', ctypes.c_int),
                ('cy', ctypes.c_int),
                ('score', ctypes.c_float)]


app=Flask(__name__)
lib = ctypes.cdll.LoadLibrary("./libcore.so")
lib.findTarget.restype = ctypes.POINTER(ToPythonRect)
lib.measureTarget.restype = XYZ
lib.useOCR.restype = ctypes.POINTER(RetBox)


if(not os.path.exists("render_frames")):
    os.mkdir("render_frames")

if(not os.path.exists("icons")):
    os.mkdir("icons")

if(not os.path.exists("frames")):
    os.mkdir("frames")


def to_c_string(data):
    return ctypes.create_string_buffer(data.encode('gbk'), len(data))


def downLoadToClientData(fileFullPath):
    with open(fileFullPath, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    return image_data


"""
app route response

"""


@app.route('/initCamera', methods=['POST'])
def init():
    camera_id = request.form['camera_id']

    ret = lib.initCamera(camera_id, to_c_string("none"), 1)

    if ret == 1 : return jsonify(INFO="vision init success")
    elif ret == -1 : return jsonify(INFO="vision init failed")
    elif ret == -2 : return jsonify(INFO="vision init has been made")
    elif ret == -6 : return jsonify(INFO="there is no config file")


@app.route('/initOCR', methods=['POST'])
def initOCR():
    ret = lib.initOCR()

    if ret == 1 : return jsonify(INFO="ocr init success")
    elif ret == -1 : return jsonify(INFO="ocr init failed")
    elif ret == -2 : return jsonify(INFO="ocr init has been made")


@app.route('/loadIcon', methods=['POST'])
def load_icon():
    keyFn = request.form['keyFn']
    iconPath = "icons/" + keyFn
    scale = request.form['scale']
    Volume = request.form['Volume']

    img = request.files.get('iconFile')
    img.save(iconPath)

    if not os.path.exists(iconPath):
        return jsonify(INFO="there is no icon file named {}, upload failed".format(keyFn))

    ret = lib.uploadIcon(to_c_string(iconPath), int(scale), int(Volume))

    if ret == -3 : return jsonify(INFO="upload icons is full")
    elif ret == -4: return jsonify(INFO="the icon has been in progress")
    else : return jsonify(INFO="success there is {} in progress".format(ret))


@app.route('/findIcon', methods=['POST'])
def find_icon():
    iconName =request.form['iconName']
    OV = request.form['OV']
    saveShow =request.form['saveShow']

    iconPath = "icons/" + iconName

    rectS = lib.findTarget(to_c_string(iconPath), c_float(float(OV)), int(saveShow))
    ret = []
    idx = 0
    while 1:
        if rectS[idx].x == -1000:
            break
        ret.append([rectS[idx].x, rectS[idx].y, rectS[idx].w, rectS[idx].h])
        idx += 1

    if int(saveShow):
        return jsonify(detRet=ret, image=downLoadToClientData("render_frames/temp.jpg"))
    else:
        return jsonify(detRet=ret, image="none")


@app.route('/measureXY', methods=['POST'])
def measure_xy():
    x =request.form['x']
    y =request.form['y']
    fx =request.form['fx']
    cx =request.form['cx']
    fy =request.form['fy']
    cy =request.form['cy']
    Zc =request.form['Zc']

    xyz = lib.measureTarget(int(x), int(y), c_float(float(fx)),  c_float(float(cx)), c_float(float(fy)), c_float(float(cy)), c_float(float(Zc)))

    return jsonify(xyz=[xyz.x, xyz.y, xyz.z])


@app.route('/findText', methods=['POST'])
def find_text():
    textS = lib.useOCR()
    ret = []
    idx = 0
    while 1:
        if textS[idx].charNum == -1:
            break
        
        text = ""

        for text_id in range(textS[idx].charNum):
            text += chr(textS[idx].text[text_id])
        ret.append([text, textS[idx].cx, textS[idx].cy])
        idx += 1
        
    return jsonify(INFO="success", ocrRet=ret)


@app.route('/capture', methods=['POST'])
def capture():
    lib.capture()
    return jsonify(image=downLoadToClientData("frames/temp.jpg"))


@app.route('/delAllTarget', methods=['POST'])
def del_all_target():
    lib.clearIconsInPro()
    return jsonify(INFO="delete all")


if __name__=="__main__":
    app.run(port=int(args.port),host=args.ip, debug=True, threaded = False)
