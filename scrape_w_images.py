import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from PIL import Image
from io import BytesIO

# Function to extract case presentation text and images from a given URL
def extract_case_presentation(url, img_dir):
    try:
        # Make a request to fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        html_content = response.text

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the 'Case presentation' section
        case_presentation_section = soup.find('h2', string='Case presentation')

        if case_presentation_section:
            # Get the content under the <div> with the id 'Sec2-content'
            content_div = case_presentation_section.find_next_sibling('div', id='Sec2-content')
            if content_div:
                case_presentation_text = content_div.get_text(separator='\n', strip=True)
                if len(case_presentation_text) >= 200:  # Check if text length is at least 200 characters
                    # Extract and save images
                    image_filenames = []
                    for img in content_div.find_all('img'):
                        img_url = img.get('src')
                        if img_url:
                            # Check if the URL has a scheme
                            if not img_url.startswith(('http://', 'https://')):
                                img_url = 'https:' + img_url
                            try:
                                img_response = requests.get(img_url)
                                img_response.raise_for_status()
                                image = Image.open(BytesIO(img_response.content))
                                img_filename = os.path.join(img_dir, os.path.basename(img_url))
                                image.save(img_filename)
                                image_filenames.append(img_filename)
                            except Exception as e:
                                print(f"Error downloading image {img_url}: {e}")
                    return case_presentation_text, image_filenames
                else:
                    return None, None  # Skip if content is shorter than 200 characters
            else:
                return None, None  # Skip if content div is not found
        else:
            return None, None  # Skip if case presentation section is not found
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None, None  # Skip in case of any error

# Function to extract article URLs from a search results page
def get_article_urls_from_page(page_number):
    base_url = f"https://jmedicalcasereports.biomedcentral.com/articles?searchType=journalSearch&sort=PubDate&page={page_number}"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    urls = []
    for a_tag in soup.find_all('a', itemprop='url'):
        href = a_tag.get('href')
        if href:
            full_url = f"https://jmedicalcasereports.biomedcentral.com{href}"
            urls.append(full_url)

    return urls

def save_checkpoint(all_results, start_page, end_page):
    filename = f'case_presentations_w_images_{start_page}_{end_page}.xlsx'
    df = pd.DataFrame(all_results, columns=['URL', 'Case Presentation', 'Image Filenames'])
    try:
        df.to_excel(filename, index=False)
        print(f"Checkpoint saved: {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def main():
    all_results = []
    checkpoint_interval = 5
    img_dir = 'downloaded_images'
    os.makedirs(img_dir, exist_ok=True)

    for page_number in range(20, 100):  # Loop from page 20 to 25
        print(f"Scraping page {page_number}...")
        urls = get_article_urls_from_page(page_number)
        time.sleep(1)  # Sleep for 1 second between page requests

        for url in urls:
            print(f"Processing URL: {url}")
            case_presentation_text, image_filenames = extract_case_presentation(url, img_dir)
            time.sleep(1)  # Sleep for 1 second between article requests
            if case_presentation_text:  # Only add to results if valid content is found
                all_results.append([url, case_presentation_text, ','.join(image_filenames)])

        if page_number % checkpoint_interval == 0:
            save_checkpoint(all_results, page_number - checkpoint_interval + 1, page_number)
            all_results = []

    # Save any remaining results
    if all_results:
        save_checkpoint(all_results, (page_number // checkpoint_interval) * checkpoint_interval + 1, 25)

    print("Scraping complete.")

if __name__ == "__main__":
    main()
