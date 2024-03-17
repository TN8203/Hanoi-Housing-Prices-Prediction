from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc
import pandas as pd

# Function to get URLs of individual house listings on one page
def getOnePage_All_IndividualURL(link_elements):
    """
    Extracts URLs of individual house listings from a list of elements.

    Parameters:
        link_elements (list): List of Selenium web elements containing links.

    Returns:
        list: List of URLs of individual house listings.
    """
    onePage_url_list = []
    for link in link_elements:
        href = link.get_attribute("href")
        onePage_url_list.append(href)
    return onePage_url_list

# Function to get the last page number of house listings
def getLastPage_Number(NextPage_xpath):
    """
    Retrieves the last page number from the pagination.

    Parameters:
        NextPage_xpath (str): XPath of the pagination element for next page.

    Returns:
        int: Last page number.
    """
    driver.get("https://batdongsan.com.vn/ban-nha-rieng-ha-noi?cIds=325")
    nextPage_list = driver.find_elements("xpath", NextPage_xpath)
    lastPage = int(nextPage_list[-1].text.replace(".", ""))
    print("The last Page is: ", lastPage)
    return lastPage

# Set up Chrome driver
options = webdriver.ChromeOptions()
driver = uc.Chrome()

individualURL_xpath = '//a[@class="js__product-link-for-product-id"]'
NextPage_xpath = '//a[@class="re__pagination-number"]'

lastPage = getLastPage_Number(NextPage_xpath)

links = []
# Iterate through each page to collect URLs of individual house listings
for i in range(1, lastPage+1):
    driver.get(f"https://batdongsan.com.vn/ban-nha-rieng-ha-noi/p{i}?cIds=325")
    link_elements = driver.find_elements("xpath", individualURL_xpath)
    temp_links = getOnePage_All_IndividualURL(link_elements)
    for e in temp_links:
        links.append(e)
    # Add a small delay to avoid overwhelming the server
    time.sleep(2)

driver.quit()

print("Total number of Houses find =", len(links))

# Remove duplicate URLs
links = list(set(links))

print("Total number of Houses after removing duplicates: ", len(links))

# Export to a CSV file
diction = {'href': links}
df0 = pd.DataFrame(diction)
df0.to_csv('URLs_data.csv', index=False)
