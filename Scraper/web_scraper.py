from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
import re
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showinfo
from threading import Thread
import os
from threading import Event
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
homepath = str(Path.home()/'Desktop')
dir = homepath + "\\LinkedinScraperBot"

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
begin_code = Event()
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
    
    def write_row(self, doko, message, sheet, out_sheet, output_path):
        print(str(sheet.at[doko+1, "Full Name"]) + " message")
        sheet.at[doko+1, "Corrections"] = message
        copied_row = sheet.loc[doko+1].copy()
        out_sheet.loc[len(out_sheet.index)] = copied_row
        out_sheet.to_csv(output_path, index=False)

    def login(self):
        with open(acc_path, 'r') as f:
            username = f.readline()
            username = username[:-1]
            password = f.readline()
        try:
            login_tab = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/nav/div/a[2]')))
            login_tab.click()
            user_tab = self.scwait.until(EC.element_to_be_clickable((By.ID, 'username')))
            user_tab.send_keys(username)
            pass_tab = self.scwait.until(EC.element_to_be_clickable((By.ID, 'password')))
            pass_tab.send_keys(password)
            signin_tab = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="organic-div"]/form/div[3]/button')))
            signin_tab.click()
        except:
            self.scdriver.get("https://www.linkedin.com")
            self.login()

    def run(self):
        self.scdriver.get("https://www.linkedin.com")
        begin_code.clear()
        begin_code.wait() 
        self.login()
        begin_code.clear()
        begin_code.wait() 
        for i in range (START_RANGE, NUM_ROWS - 1):
            numbers = pd.read_csv(dir + "\\numbers.csv")
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
                break

            global progress
            progress = i
            candidate_name = self.sheet.at[i+1, "Full Name"]
            company = self.sheet.at[i+1, "AccountId"]
            
            if pd.isnull(self.sheet.at[i+1, "Online Profile URL(s)"]):
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue
            url = self.sheet.at[i+1, "Online Profile URL(s)"]
            self.scdriver.get(url)

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
                            self.scdriver.refresh()
                            continue
                trying = False
            if not worked:
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue
            
            pfco = self.scwait.until(EC.element_to_be_clickable((By.XPATH, '//*[@class = "application-outlet"]/div[3]/div/div/div[2]/div/div/main/section[1]/div[2]/div[2]/ul/li[1]/button/span/div')))
            profile_company = pfco.text

            if pd.isnull(self.sheet.loc[i+1, "AccountId"]):
                self.write_row(i, profile_company, self.sheet, self.out_sheet, self.output_path)
                numbers = pd.read_csv(dir + "\\numbers.csv")
                numbers.at[i - START_RANGE, 'Status'] = 'Done'
                numbers.to_csv(dir + '\\numbers.csv', index=False)
                continue

            company_keywords = re.split('[ -;.,]', company)
            company_keywords = [element.lower() for element in company_keywords]
            web_company_keywords = re.split('[ -;.,]', profile_company)
            web_company_keywords = [element.lower() for element in web_company_keywords]

            for word in company_keywords:
                if word != 'the' and word not in web_company_keywords:
                    self.write_row(i, profile_company, self.sheet, self.out_sheet, self.output_path)
                    break
            
            numbers = pd.read_csv(dir + "\\numbers.csv")
            numbers.at[i - START_RANGE, 'Status'] = 'Done'
            numbers.to_csv(dir + '\\numbers.csv', index=False)

            self.scdriver.execute_script("window.scrollTo(0, 1080)")
            time.sleep(random.randint(1, 3))

class Scraper(Thread):
    def __init__(self):
        super().__init__()
            
    def run(self):
        temp_file = dir + "\\path_website1234abcd.html"
        driver.get(temp_file)
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
        self.pbar.grid(column=0, row=1, columnspan=2, padx=10, pady=20)
        self.create_buttons()
        self.count = 0

    def update_progress_label(self):
        self.pbar.step(self.pbar['value'])

    def create_progress_bar(self): 
        self.pbar.grid(column=0, row=1, columnspan=2, padx=10, pady=20)
        self.value_label = ttk.Label(self, text=self.update_progress_label()) 
        self.value_label.grid(column=0, row=2, columnspan=2)        

    def start(self):
        begin_code.set()
        num = START_RANGE
        self.pbar['value'] = num
    
    def increment_pbar(self):
        self.pbar['value'] = progress  / (NUM_ROWS - 2) * 100
        if self.pbar['value'] >= 99.9:
            showinfo(message='Scraping completed!')

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

def win_thread_func():
    window = UserWindow()
    window.mainloop()

if __name__ == "__main__":
    # creates scraper worker threads and runs concurrently
    worker_thread = Scraper()
    window_thread = Thread(target=win_thread_func)
    worker_thread.start()
    window_thread.start()
    worker_thread.join()
    window_thread.join()
