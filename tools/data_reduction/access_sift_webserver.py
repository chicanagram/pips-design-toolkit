#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:28:26 2024

@author: charmainechia
"""

def main(argv=None):
    """Fill out SIFT webserver"""
    import os
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import ChromiumOptions
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import pandas as pd

    msa_path = argv['-a']
    filepath = argv['--fname']

    # set up web driver
    website = 'https://sift.bii.a-star.edu.sg/www/SIFT_aligned_seqs_submit.html'
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # get website
    driver.get(website)
    time.sleep(3)

    # upload fasta file
    file = driver.find_element(By.NAME, "alignment_file")
    file.send_keys(r"{}".format(msa_path))
    print('Uploaded fasta msa.')

    # click submit
    submit_button = driver.find_element(By.ID, 'submit_button')
    submit_button.click()
    print('Clicked Submit.')
    time.sleep(3)

    # wait for next page to load and click "Scaled Probabilities for Entire Protein" button
    res_link = driver.find_element(By.PARTIAL_LINK_TEXT, 'Scaled Probabilities')
    res_link.click()

    # switch to new html page and download data
    wait = WebDriverWait(driver, 2)
    wait.until(EC.number_of_windows_to_be(2))
    driver.switch_to.window(driver.window_handles[1])
    url = driver.current_url
    print('Obtained results.')

    # get table data
    df = pd.read_html(url)
    df = df[0]
    df = df.dropna(axis=0)
    df = df[df.pos!='pos']
    # parse 'pos' column in to 'pos', 'wt', 'prob' columns
    pos_wt_prob_list = df['pos'].tolist()
    pos_list = []
    wt_list = []
    prob_list = []
    for pos_wt_prob in pos_wt_prob_list:
        pos_wt_prob_lst = pos_wt_prob.split()
        print(len(pos_wt_prob_lst), pos_wt_prob_lst)
        pos_wt, prob = pos_wt_prob_lst[0], pos_wt_prob_lst[1]
        pos = int(pos_wt[:-1])
        wt = pos_wt[-1]
        pos_list.append(pos)
        wt_list.append(wt)
        prob_list.append(float(prob))

    # add columns
    df['pos'] = pos_list
    df.insert(loc=1, column='wt', value=wt_list)
    df.insert(loc=2, column='prob', value=prob_list)
    df.to_csv(filepath, index=False)
    print(df)
    print(f'Saved sift results to {filepath}')

    time.sleep(5)
    driver.quit()
    return df


if __name__ == '__main__':
    main()