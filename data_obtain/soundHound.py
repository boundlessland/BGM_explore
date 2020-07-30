# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:11:02 2020

@author: jsx
"""

from appium import webdriver
from time import sleep
import csv
import codecs
import datetime as dt
import time as localTime
import pandas as pd


class Action:

    infoList = []

    def __init__(self):
        self.desired_caps = {
            "platformName": "Android",
            "deviceName": "127.0.0.1:62001"
        }
        self.server = 'http://localhost:4723/wd/hub'
        self.driver = webdriver.Remote(self.server, self.desired_caps)

    def miniRecognise(self):#用于粗略判断视频是否存在BGM
        global cnt
        global index
        dataFrame = pd.read_csv('')
        for i in range(index, len(dataFrame)):
            cnt = i
            url = dataFrame.iloc[i]['video_url']
            self.driver.set_clipboard_text(url)
            sleep(5)
            try:
                bgm = self.driver.find_element_by_xpath('/hierarchy/android.widget.FrameLayout/android.widget.FrameLayout/android.widget.FrameLayout/android.widget.FrameLayout/android.widget.LinearLayout/android.widget.RelativeLayout/android.widget.RelativeLayout/android.widget.TextView[2]').text
                dataFrame.loc[i, 'bgm'] = bgm
            except:
                dataFrame.to_csv('')
            self.driver.tap([(520, 360), (536, 373)], 500)

    def main(self):
        self.miniRecognise()


if __name__ == '__main__':
    action = Action()
    action.main()