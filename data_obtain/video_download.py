from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time
import pyautogui
import json
import os

def isElementPresent(self,by,value):
    #从selenium.common.exceptions 模块导入 NoSuchElementException类
    from selenium.common.exceptions import NoSuchElementException
    try:
        element = self.driver.find_element(by = by, value= value)
    #原文是except NoSuchElementException, e:
    except NoSuchElementException as e:
        #打印异常信息
        print(e)
        #发生了NoSuchElementException异常，说明页面中未找到该元素，返回False
        return False
    else:
        #没有发生异常，表示在页面中找到了该元素，返回True
        return True

for j in range(1,16):
    nm = str(j) + '.json'
    with open(nm,'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    id = 0
    succeed_list = []
    error_list = []

    for i in json_data['data']:

        url = i['aweme']['aweme_url']
        aweme_id = i['aweme']['aweme_id']

        wd = webdriver.Chrome('chromedriver.exe')
        wd.get(url)

        element = wd.find_element_by_class_name('play-btn')

        ac1 = ActionChains(wd)
        ac1.move_to_element(element)
        ac1.click()
        try:
            ac1.perform()

            time.sleep(1)

            ac2 = ActionChains(wd)
            element = wd.find_element_by_class_name('player')
            ac2.move_to_element(element)
            ac2.context_click()
            ac2.perform()

            time.sleep(1)
            pyautogui.typewrite(['down', 'down', 'down', 'down'])  # 选中右键菜单中第2个选项
            pyautogui.typewrite(['enter'])  # 最后一个按键： mac电脑用的return，Windows应用enter

            time.sleep(5)
            pyautogui.moveTo(765, 565)
            pyautogui.click()
            pyautogui.typewrite(['enter'])
            time.sleep(5)

            print(id)
            print('succeed')
            succeed_list.append(id)

            src_path = 'C:\\Users\\James\\Downloads\\下载.mp4'
            des_path = 'C:\\Users\\James\\Documents\\抖音下载\\' + str(j) + '\\' + aweme_id + '.wav'

            os.rename(src_path, des_path)

        except Exception:
            print(id)
            print('error')
            error_list.append(id)

        wd.close()
        id += 1

print(0)


