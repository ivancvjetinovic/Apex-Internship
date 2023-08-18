from typing import Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
import re
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showinfo
from threading import Thread
from tkinter.messagebox import showerror
import os
from threading import Event
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
homepath = str(Path.home()/'Desktop')
dir = homepath + "\\LinkedinScraperBot"

# Required Information
text_path = ""

option = Options()
option.add_argument("--disable-infobars")
option.add_argument("start-maximized")
option.add_argument("--disable-extensions")
option.add_experimental_option(
    "prefs", {"profile.default_content_setting_values.notifications": 1}
)
option.add_argument("--incognito")
service = ChromeService(executable_path="\\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=option)
wait = WebDriverWait(driver, random.randint(2, 5))
driver.delete_all_cookies()

START_RANGE = 0

# event used to start scraping
begin_code = Event()

# modifies progress bar length
progress = 0

NUM_ROWS = 100
acc_path = dir + "\\account1234abcd.txt"
locks = set()

class ScraperInstance(Thread):
    def __init__(self, sheet, out_sheet, output_path, ):
        super().__init__()
        self.sheet = sheet
        self.out_sheet = out_sheet
        self.output_path = output_path
        option.add_argument("--window-size=300,400")
        self.scdriver = webdriver.Chrome(service=service, options=option)
        self.scwait = WebDriverWait(self.scdriver, random.randint(2, 5))
        self.scdriver.delete_all_cookies()
    
    # helper function for writing corrections to csv
    def write_row(self, doko, message, sheet, out_sheet, output_path):
        print(str(sheet.at[doko+1, "Full Name"]) + " message")
        sheet.at[doko+1, "Corrections"] = message
        copied_row = sheet.loc[doko+1].copy()
        out_sheet.loc[len(out_sheet.index)] = copied_row
        out_sheet.to_csv(output_path, index=False)

    # logs into dummy account
    def login(self):
        # acc_path = self.find_site("account1234abcd.txt")
        # account details
        with open(acc_path, 'r') as f:
            username = f.readline()
            username = username[:-1]
            password = f.readline()
        # old users "poepukon@gmail.com" # "aisukreem8@gmail.com" # "waterbotl39@gmail.com"
        # old passwords "Popcorn!1" # "aisukreem$1$" # for email -> "icecream8!" # "waterbottle1"
        # signs in (slowly)
        # clears cookies       
        # time.sleep(random.randint(10, 15))
        actions = ActionChains(self.scdriver)
        actions.key_down(Keys.CONTROL)
        actions.send_keys(Keys.F5)
        actions.key_up(Keys.CONTROL)
        actions.perform()
        time.sleep(random.randint(8, 23))
        try:
            login_tab = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/nav/div/a[2]')))
            login_tab.click()
            time.sleep(random.randint(3, 7))
            user_tab = self.scwait.until(EC.element_to_be_clickable((By.ID, 'username')))
            user_tab.send_keys(username)
            time.sleep(random.randint(3, 7))
            pass_tab = self.scwait.until(EC.element_to_be_clickable((By.ID, 'password')))
            pass_tab.send_keys(password)
            time.sleep(random.randint(3, 7))
            signin_tab = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="organic-div"]/form/div[3]/button')))
            signin_tab.click()
        except:
            self.scdriver.get("https://www.linkedin.com")
            self.login()
        # time.sleep(random.randint(17,110))
        # time.sleep(random.randint(5, 15))

    def run(self):
        # calls website login function
        self.scdriver.get("https://www.linkedin.com")
        begin_code.clear()
        begin_code.wait() 

        self.login()
        begin_code.clear()
        begin_code.wait() 
        time.sleep(random.randint(3, 46))
        # main loop - go through all candidates
        for i in range (START_RANGE, NUM_ROWS - 1):
            numbers = pd.read_csv(dir + "\\numbers.csv")
            # change i to correct value
            global locks
            done = True
            for j in range(numbers.shape[0]):
                num = START_RANGE + j
                if (numbers.at[j, 'Status'] == 'Not') and (num not in locks):
                    locks.add(num)
                    i = num
                    print(locks)
                    done = False
                    break

            if done:
                print("Fully Done")
                break
            
            global progress
            progress = i

            # indexes csv file for candidate information
            candidate_name = self.sheet.at[i+1, "Full Name"]
            company = self.sheet.at[i+1, "AccountId"]

            # handles empty linkedin url cell
            if pd.isnull(self.sheet.at[i+1, "Online Profile URL(s)"]):
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue
            url = self.sheet.at[i+1, "Online Profile URL(s)"]

            # opens candidate linkedin
            self.scdriver.get(url)
            # time.sleep(random.randint(20, 60))
            time.sleep(random.randint(2, 6))

            # try catch looping variables
            trying = True
            worked = True
            count = 0
            while trying:
                try:
                    # try selecting company name in bio
                    pfco = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '//*[@class = "application-outlet"]/div[3]/div/div/div[2]/div/div/main/section[1]/div[2]/div[2]/ul/li[1]/button/span/div')))
                except:
                    # if webiste does not exist
                    if self.scdriver.current_url == "https://www.linkedin.com/404/":
                        worked = False
                    else:
                        # reports if needs manual work
                        if count:
                            worked = False
                            self.write_row(i, "Manual", self.sheet, self.out_sheet, self.output_path)
                        else:
                            # reloads page once to account for failed load
                            count += 1
                            time.sleep(random.randint(13,40))
                            self.scdriver.refresh()
                            time.sleep(random.randint(17,42))
                            continue
                trying = False
            # if candidate was not found
            if not worked:
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue
            
            # profile company name
            pfco = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '//*[@class = "application-outlet"]/div[3]/div/div/div[2]/div/div/main/section[1]/div[2]/div[2]/ul/li[1]/button/span/div')))
            profile_company = pfco.text

            # handles edge case where cell is empty
            if pd.isnull(self.sheet.loc[i+1, "AccountId"]):
                self.write_row(i, profile_company, self.sheet, self.out_sheet, self.output_path)
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue

            # divides up companies into two separate lists and makes both lower case
            company_keywords = re.split('[ -;.,]', company)
            company_keywords = [element.lower() for element in company_keywords]
            web_company_keywords = re.split('[ -;.,]', profile_company)
            web_company_keywords = [element.lower() for element in web_company_keywords]

            # compares first value in each list unless is 'the'
            # thereby compares if equal (cases follow a first word equal pattern)
            for word in company_keywords:
                if word == 'the':
                    continue
                if word not in web_company_keywords:
                    self.write_row(i, profile_company, self.sheet, self.out_sheet, self.output_path)
                break
            
            # keep for debugging purpose
            print(candidate_name + " Done!")
            numbers = pd.read_csv(dir + "\\numbers.csv")
            numbers.at[i - START_RANGE, 'Status'] = 'Done'
            numbers.to_csv(dir + '\\numbers.csv', index=False)

            # sleep ~5 mins total and scroll a tiny bit
            time.sleep(random.randint(225, 375))
            self.scdriver.execute_script("window.scrollTo(0, 1080)")
            time.sleep(random.randint(300, 457))

class Scraper(Thread):
    def __init__(self):
        # initializes parent Thread
        super().__init__()
            
    def run(self):
        # opens custom website
        temp_file = dir + "\\path_website1234abcd.html"
        print(temp_file)
        time.sleep(1)

        driver.get(temp_file)

        # wait for button to be clicked
        begin_code.wait()
        storage_paths = dir + "\\stored_paths1234abcd.txt"
        
        if driver.find_element(By.ID, "pathStatus").text == "New":
            iwait = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'input')))
            itext = iwait.text
            owait = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'output')))
            otext = owait.text
            nwait = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'text')))
            ntext = nwait.text

            open(storage_paths, 'w').close()

            with open(storage_paths, 'a') as f:
                f.write(itext + "\n")
                f.write(otext + "\n")
                f.write(ntext)

        with open(storage_paths, 'r') as f:
                input_path = f.readline()
                input_path = input_path[:-1]
                output_path = f.readline()
                output_path = output_path[:-1]
                tpath = f.readline()

        if input_path[0] == '"':
            input_path = input_path[1:-1]
        if output_path[0] == '"':
            output_path = output_path[1:-1]
        if tpath[0] == '"':
            tpath = tpath[1:-1]
        
        global text_path 
        text_path = tpath

        with open(text_path, 'r') as f:
            global START_RANGE
            START_RANGE = int(f.read())
        print("read ", START_RANGE)        
        sheet = pd.read_csv(input_path)
        out_sheet = pd.read_csv(output_path)

        # constant num_rows for looping
        global NUM_ROWS
        NUM_ROWS = sheet.shape[0]
        
        if driver.find_element(By.ID, "loginStatus").text == "New":
            utemp = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'username')))
            ptemp = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'password')))
            user = utemp.text
            passw = ptemp.text

            open(acc_path, 'w').close()

            with open(acc_path, 'a') as f:
                f.write(user + "\n")
                f.write(passw)

        data = []
        
        for i in range (NUM_ROWS - START_RANGE - 1):
            data.append(['Not'])
        numbers = pd.DataFrame(data, columns=['Status'])
        numbers.to_csv(dir + "\\numbers.csv", index=False)
        
        window_count = int(driver.find_element(By.CLASS_NAME, "selected-number").text)
        driver.close()
        ###
        if (window_count >= 1):
            sc1 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 2):
            sc2 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 3):
            sc3 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 4):
            sc4 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 5):
            sc5 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 6):
            sc6 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 7):
            sc7 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 8):
            sc8 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 9):
            sc9 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 10):
            sc10 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 11):
            sc11 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 12):
            sc12 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 13):
            sc13 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 14):
            sc14 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 15):
            sc15 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 16):
            sc16 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 17):
            sc17 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 18):
            sc18 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 19):
            sc19 = ScraperInstance(sheet, out_sheet, output_path)
        if (window_count >= 20):
            sc20 = ScraperInstance(sheet, out_sheet, output_path)
        ####
        if (window_count >= 1):
            sc1.start()
        if (window_count >= 2):
            sc2.start()
        if (window_count >= 3):
            sc3.start()
        if (window_count >= 4):
            sc4.start()
        if (window_count >= 5):
            sc5.start()
        if (window_count >= 6):
            sc6.start()
        if (window_count >= 7):
            sc7.start()
        if (window_count >= 8):
            sc8.start()
        if (window_count >= 9):
            sc9.start()
        if (window_count >= 10):
            sc10.start()
        if (window_count >= 11):
            sc11.start()
        if (window_count >= 12):
            sc12.start()
        if (window_count >= 13):
            sc13.start()
        if (window_count >= 14):
            sc14.start()
        if (window_count >= 15):
            sc15.start()
        if (window_count >= 16):
            sc16.start()
        if (window_count >= 17):
            sc17.start()
        if (window_count >= 18):
            sc18.start()
        if (window_count >= 19):
            sc19.start()
        if (window_count >= 20):
            sc20.start()
        ####
        if (window_count >= 1):
            sc1.join()
        if (window_count >= 2):
            sc2.join()
        if (window_count >= 3):
            sc3.join()
        if (window_count >= 4):
            sc4.join()
        if (window_count >= 5):
            sc5.join()
        if (window_count >= 6):
            sc6.join()
        if (window_count >= 7):
            sc7.join()
        if (window_count >= 8):
            sc8.join()
        if (window_count >= 9):
            sc9.join()
        if (window_count >= 10):
            sc10.join()
        if (window_count >= 11):
            sc11.join()
        if (window_count >= 12):
            sc12.join()
        if (window_count >= 13):
            sc13.join()
        if (window_count >= 14):
            sc14.join()
        if (window_count >= 15):
            sc15.join()
        if (window_count >= 16):
            sc16.join()
        if (window_count >= 17):
            sc17.join()
        if (window_count >= 18):
            sc18.join()
        if (window_count >= 19):
            sc19.join()
        if (window_count >= 20):
            sc20.join()

class UserWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Company Comparator")
        self.geometry("300x170")
        self.resizable(False, False)
        self.pbar = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280
        )
        # places pbar onto window
        self.pbar.grid(column=0, row=1, columnspan=2, padx=10, pady=20)
        self.create_buttons()
        self.count = 0

    # updates progress bar
    def update_progress_label(self):
        self.pbar.step(self.pbar['value'])

    def create_progress_bar(self): 
        self.pbar.grid(column=0, row=1, columnspan=2, padx=10, pady=20)
        self.value_label = ttk.Label(self, text=self.update_progress_label()) 
        self.value_label.grid(column=0, row=2, columnspan=2)        

    # start button
    def start(self):
        # self.count += 1
        begin_code.set()
        num = START_RANGE #)) / NUM_ROWS * 100
        self.pbar['value'] = num
        # disable clicking start after program is started
        # if self.count == 3:
        #     self.start_button['state'] = tk.DISABLED
        # showinfo(message='Feel free to minimze tab. Do not interact with the webpage.')
    
    def increment_pbar(self):
        self.pbar['value'] = progress  / (NUM_ROWS - 2) * 100
        # showinfo(message=str(NUM_ROWS-progress-2)+' left')
        if self.pbar['value'] >= 99.9:
            showinfo(message='Scraping completed!')
            # self.update_button['state'] = tk.DISABLED

    # stop button
    def stop(self):
        numbers = pd.read_csv(dir + "\\numbers.csv")
        val = 0
        for i in range(numbers.shape[0]):
            if numbers.at[i, 'Status'] == 'Not':
                val = START_RANGE + i
                break
        global text_path
        print("Stopping", text_path)
        if len(text_path):
            f = open(text_path, 'w')
            f.write(str(val))
            f.close()
        os._exit(1)
    
    def create_buttons(self):
        self.start_button = ttk.Button(
            self,
            text='Start',
            command=lambda : self.start()
        )
        self.start_button.grid(column=0, row=0, padx=10, pady=10, sticky=tk.E)
        self.stop_button = ttk.Button(
            self,
            text='Stop',
            command=lambda : self.stop()
        )
        self.stop_button.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        self.update_button = ttk.Button(
            self,
            text='Update Progress Bar',
            command=lambda : self.increment_pbar()
        )
        self.update_button.grid(column=0, row=4, padx=10, pady=10, sticky=tk.S)

# function that runs the tkinter gui thread
def win_thread_func():
    window = UserWindow()
    window.mainloop()

if __name__ == "__main__":
    # creates scraper worker threads and runs concurrently
    worker_thread = Scraper()
    window_thread = Thread(target=win_thread_func) # args=(event)
    worker_thread.start()
    window_thread.start()
    worker_thread.join()
    window_thread.join()
