import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd
from datetime import datetime

import requests

# specify the url
urlpage = "https://www.twitch.tv/gamesdonequick"
print(urlpage)
# run firefox webdriver from executable path of your choice
driver = webdriver.Firefox()

# get web page
driver.get(urlpage)
# execute script to scroll down the page
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
# sleep for 30s
time.sleep(5)

# create empty array to store data
data = []

# get viewer div
viewer_container = driver.find_elements_by_xpath('//*[@id="root"]/div/div[2]/div[2]/main/div[2]/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div[1]/div/div[1]/div[2]/div/div/div[1]/div/div[1]/div[2]')

viewers = viewer_container[0].text
print(viewers)

# get category
category_container = driver.find_elements_by_xpath('//*[@id="root"]/div/div[2]/div[2]/main/div[2]/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div[1]/div/div[2]/div[2]/div[1]/div[1]/p/a')
category = category_container[0].text
print(category)



driver.quit()