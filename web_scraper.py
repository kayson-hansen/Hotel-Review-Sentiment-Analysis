import csv
from selenium import webdriver  # web scraper package
import time

# The paths to the files to store the hotel review data
file_paths = [
    "/users/kaysonhansen/cs129/VenetianHotelReviews.csv",
    "/users/kaysonhansen/cs129/MirageHotelReviews.csv",
    "/users/kaysonhansen/cs129/MandalayBayHotelReviews.csv",
    "/users/kaysonhansen/cs129/TrumpInternationalHotelReviews.csv",
    "/users/kaysonhansen/cs129/LuxorHotelReviews.csv",
    "/users/kaysonhansen/cs129/TreasureIslandHotelReviews.csv",
    "/users/kaysonhansen/cs129/ParisHotelReviews.csv",
    "/users/kaysonhansen/cs129/CaesarsPalaceHotelReviews.csv",
    "/users/kaysonhansen/cs129/ARIAHotelReviews.csv",
    "/users/kaysonhansen/cs129/PlanetHollywoodHotelReviews.csv",
    "/users/kaysonhansen/cs129/PalazzoHotelReviews.csv",
    "/users/kaysonhansen/cs129/ParkMGMReviews.csv",
    "/users/kaysonhansen/cs129/VdaraHotelReviews.csv",
    "/users/kaysonhansen/cs129/ExcaliburHotelReviews.csv",
    "/users/kaysonhansen/cs129/WynnHotelReviews.csv",
    "/users/kaysonhansen/cs129/RioHotelReviews.csv"
]

# There are 10 reviews per page, giving a total of 10 * pages_to_scrape reviews per hotel
pages_to_scrape = 2000

# List of urls for hotels in Las Vegas with at least 10,000 reviews
urls = [
    "https://www.tripadvisor.com/Hotel_Review-g45963-d97704-Reviews-The_Venetian_Resort-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d97737-Reviews-The_Mirage_Hotel_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91886-Reviews-Mandalay_Bay_Resort_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d1022061-Reviews-Trump_International_Hotel_Las_Vegas-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d111709-Reviews-Luxor_Hotel_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91967-Reviews-Treasure_Island_TI_Hotel_Casino_a_Radisson_Hotel-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d143336-Reviews-Paris_Las_Vegas-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91762-Reviews-Caesars_Palace-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91925-Reviews-ARIA_Resort_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91687-Reviews-Planet_Hollywood_Las_Vegas_Resort_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d675000-Reviews-The_Palazzo_at_The_Venetian-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d97712-Reviews-Park_MGM_Las_Vegas-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d1474086-Reviews-Vdara_Hotel_Spa-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d97786-Reviews-Excalibur_Hotel_Casino-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d503598-Reviews-Wynn_Las_Vegas-Las_Vegas_Nevada.html",
    "https://www.tripadvisor.com/Hotel_Review-g45963-d91673-Reviews-Rio_All_Suite_Hotel_Casino-Las_Vegas_Nevada.html"
]

# n = which hotel to scrape
# 0 = The Venetian, 1 = The Mirage, 2 = Mandalay Bay, 3 = Trump International, 4 = Luxor,
# 5 = Treasure Island, 6 = Paris, 7 = Caesars Palace, 8 = ARIA, 9 = Planet Hollywood,
# 10 = Palazzo, 11 = Park MGM, 12 = Vdara, 13 = Excalibur, 14 = Wynn, 15 = Rio
n = 15

# import the webdriver
driver = webdriver.Chrome()
driver.get(urls[n])

# open the file to save the review
csvFile = open(file_paths[n], 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)

for i in range(0, pages_to_scrape):

    # give the DOM time to load (3 seconds)
    time.sleep(3)

    # Click the "expand review" link to reveal the entire review.
    driver.find_element("xpath",
                        ".//div[contains(@data-test-target, 'expand-review')]").click()

    # Find all reviews in the current page and store them all to a container
    container = driver.find_elements("xpath", "//div[@data-reviewid]")

   # Parse each review in the container
    for j in range(len(container)):  # A loop defined by the number of reviews

        # Grab the rating
        rating = container[j].find_element("xpath",
                                           ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]

        """
        # Grab the title
        title = container[j].find_element("xpath",
                                          ".//div[contains(@data-test-target, 'review-title')]").text
        """

        # Grab the review text, removing newlines and commas
        review = container[j].find_element("xpath",
                                           ".//q[@class='QewHA H4 _a']").text.replace("\n", "  ").replace(",", "")

        # Write review data to the csv
        csvWriter.writerow([rating, review])

    # When all the reviews in the container have been processed, move to the next page and repeat
    driver.find_element("xpath",
                        './/a[contains(@class, "ui_button nav next primary ")]').click()

# When all pages have been processed, quit the driver
driver.quit()
