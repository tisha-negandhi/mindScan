import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin

        
def scrape_and_run1():
    blogs = []
    page = requests.get("https://www.healthline.com/content-series/youre-not-alone")
    soup = bs(page.content, 'html.parser')
    articles = soup.find_all('div', class_='css-8sm3l3')
    for article in articles:
        # Extracting author name
        author_span = article.find('section', class_='css-rwbuqt')
        if author_span:
            author_name = author_span.text.split("By ")[1]
        else:
            author_span = article.find('section', class_='css-1we139v')
            author_name = author_span.text.split("By ")[1]
        # Extracting title
        title_link = article.find('a', class_='css-16e3huk')
        title = title_link.text if title_link else "Unknown Title"

        # Extracting image URL
        image = article.find('lazy-image')
        image_url = image['src'] if image else "No Image URL"

        # Extracting article link
        link = title_link['href'] if title_link else "No Link"
        blogs.append(
            {"title": title, "author": author_name, "image": image_url, "link":link})

        # Printing the results
    print(len(blogs))
    return blogs


def scrape_and_run2():
    blogs = []
    page = requests.get("https://www.healthline.com/content-series/diagnosis-diaries")
    soup = bs(page.content, 'html.parser')
    articles = soup.find_all('div', class_='css-8sm3l3')
    for article in articles:
        # Extracting author name
        author_span = article.find('section', class_='css-rwbuqt')
        if author_span:
            author_name = author_span.text.split("By ")[1]
        else:
            author_span = article.find('section', class_='css-1we139v')
            author_name = author_span.text.split("By ")[1]
        # Extracting title
        title_link = article.find('a', class_='css-16e3huk')
        title = title_link.text if title_link else "Unknown Title"

        # Extracting image URL
        image = article.find('lazy-image')
        image_url = image['src'] if image else "No Image URL"

        # Extracting article link
        link = title_link['href'] if title_link else "No Link"
        print("Author:", author_name)
        print("Title:", title)
        print("Image URL:", image_url)
        print("Link:", link)
        print("\n")
        blogs.append(
            {"title": title, "author": author_name, "image": image_url, "link":link})
    print(len(blogs))

        # Printing the results
    return blogs




def scrape_and_run3():
    blogs = {}
    widget_ids = ["widget-small-listing-0", "widget-small-listing-2","widget-small-listing-3", "widget-small-listing-4", "widget-small-listing-6" ]
    page = requests.get("https://www.healthline.com/reviews/mental-health-services-and-products")
    soup = bs(page.content, 'html.parser')
    for widget_id in widget_ids:
        articles = soup.find_all('div', {'data-widget-id': widget_id})
        for article in articles:
            main_title = article.find('h2').text.strip()
            li_elements = article.find_all('li', class_='css-1ib8oek')
            for article in li_elements:
                title = article.find('a', {'data-testid': 'title-link'}).text.strip() if article.find('a', {'data-testid': 'title-link'}) else None

                text = article.find('p', class_='css-ghc6ug').text.strip() if article.find('p', class_='css-ghc6ug') else None

                image_link = article.find('lazy-image')['src'] if article.find('lazy-image') else None

                page_link = article.find('a', {'data-testid': 'title-link'}).get('href') if article.find('a', {'data-testid': 'title-link'}) else None
                if title is not None or text is not None or image_link is not None or page_link is not None:
                    if main_title not in blogs:
                        blogs[main_title] = []
                    blogs[main_title].append({
                        "Title": title,
                        "Text": text,
                        "Image": image_link,
                        "Link": page_link
                    })
    return blogs

def scrape_and_run4():
    blogs = []
    page = requests.get("https://www.healthline.com/reviews/mental-health-services-and-products")
    soup = bs(page.content, 'html.parser')
    articles = soup.find_all('ul', class_='css-1q1zlz3')
    for article in articles:
        li_elements = article.find_all('li', class_='css-1ib8oek')
        for li in li_elements:
            title = li.find('a', {'data-testid': 'title-link'}).text.strip() if li.find('a', {'data-testid': 'title-link'}) else None
            text = li.find('p', class_='css-ghc6ug').text.strip() if li.find('p', class_='css-ghc6ug') else None
            image_link = li.find('lazy-image')['src'] if li.find('lazy-image') else None
            page_link = li.find('a', {'data-testid': 'title-link'}).get('href') if li.find('a', {'data-testid': 'title-link'}) else None
            if title is not None or text is not None or image_link is not None or page_link is not None:
                if "support-group" in page_link:
                    blogs.append(
            {"Title": title, "Text": text, "Image": image_link, "Link":page_link})
    return blogs
